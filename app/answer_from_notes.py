"""Answer user-supplied questions using retrieved lecture notes (RAG).

Students paste or upload their own questions (e.g. past papers). For each
question we BM25-retrieve note chunks and ask Claude for a short outline plus
a fuller model answer grounded only in those excerpts.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from .generator import MODEL, GenerationError, _client
from .retriever import BM25Retriever

MAX_QUESTIONS = 15
TOP_K_CHUNKS = 6
MAX_WORDS_PER_CHUNK = 140

# Split pasted/file text into individual questions.
_Q_START = re.compile(
    r"(?m)^(?:\s*(?:Q(?:uestion)?\s*)?(\d+)[\.\)\:]\s+|(?:Question\s+(\d+)\s*[:.\-–—]\s*))",
    re.IGNORECASE,
)


class NoteAnswer(BaseModel):
    question: str
    outline: list[str] = Field(default_factory=list)
    full_answer: str = ""
    source_quotes: list[str] = Field(default_factory=list)
    notes_cover_question: bool = True


class _AnswerOut(BaseModel):
    outline: list[str] = Field(default_factory=list)
    full_answer: str = ""
    source_quotes: list[str] = Field(default_factory=list)
    notes_cover_question: bool = True


def parse_questions(text: str) -> list[str]:
    """Split free-form past-question text into individual questions."""
    if not text or not text.strip():
        return []

    raw = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Normalise fancy bullets
    raw = re.sub(r"[•●▪]", "-", raw)

    parts: list[str] = []
    matches = list(_Q_START.finditer(raw))
    if matches:
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            body = raw[start:end].strip()
            # Include leading number for clarity when useful
            num = m.group(1) or m.group(2)
            q = body
            if num and not body.lower().startswith("question"):
                q = body
            if q:
                parts.append(_clean_question(q))
    else:
        # Blank-line separated blocks, else non-empty lines
        blocks = re.split(r"\n\s*\n+", raw)
        if len(blocks) > 1:
            parts = [_clean_question(b) for b in blocks if _clean_question(b)]
        else:
            parts = [
                _clean_question(line)
                for line in raw.split("\n")
                if _clean_question(line) and len(_clean_question(line)) > 8
            ]

    # Deduplicate (case-insensitive)
    seen: set[str] = set()
    out: list[str] = []
    for q in parts:
        key = re.sub(r"\s+", " ", q.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _clean_question(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    # Drop pure marks lines
    if re.fullmatch(r"[\d\s\.\(\)marksMarks]+", s):
        return ""
    if len(s) < 8:
        return ""
    return s


def _trim(chunk: str) -> str:
    words = chunk.split()
    if len(words) <= MAX_WORDS_PER_CHUNK:
        return chunk.strip()
    return " ".join(words[:MAX_WORDS_PER_CHUNK]).strip() + "…"


def _answer_one(
    *,
    question: str,
    chunks: list[str],
    doc_title: str,
) -> NoteAnswer:
    import anthropic

    sources = "\n\n".join(
        f"[Source {i + 1}]\n{c}" for i, c in enumerate(chunks)
    ) or "(No relevant excerpts found in the notes.)"

    task = f"""Answer this student question using ONLY the lecture excerpts below.

Document: "{doc_title}"

Question:
{question}

Lecture excerpts:
{sources}

Rules:
- If the excerpts do not contain enough information, set notes_cover_question=false
  and say what is missing in full_answer; keep outline brief.
- outline: 3–6 short bullet points (marking-scheme style).
- full_answer: a clear model answer in 1–4 short paragraphs (not a novel).
- source_quotes: 0–3 short verbatim phrases from the excerpts (empty if none).
- Do not invent course content that is not supported by the excerpts.
"""

    client = _client()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=4000,
            system=(
                "You are a careful university tutor. You answer only from provided "
                "lecture notes. Prefer clarity and exam usefulness over length."
            ),
            messages=[{"role": "user", "content": task}],
            output_format=_AnswerOut,
        )
    except anthropic.AuthenticationError:
        raise GenerationError("The Anthropic API key was rejected.")
    except anthropic.RateLimitError:
        raise GenerationError("Rate limited. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        raise GenerationError(f"Claude API error ({e.status_code}). Try again shortly.")
    except anthropic.APIConnectionError:
        raise GenerationError("Could not reach the Claude API.")
    except Exception as e:
        raise GenerationError(f"Answer generation failed: {e}") from e

    parsed = response.parsed_output
    if parsed is None:
        raise GenerationError("Could not parse the model answer. Try again.")

    return NoteAnswer(
        question=question,
        outline=[x.strip() for x in (parsed.outline or []) if x and str(x).strip()][:8],
        full_answer=(parsed.full_answer or "").strip(),
        source_quotes=[
            x.strip() for x in (parsed.source_quotes or []) if x and str(x).strip()
        ][:4],
        notes_cover_question=bool(parsed.notes_cover_question),
    )


def answer_questions(
    *,
    questions: list[str],
    chunks: list[str],
    retriever: BM25Retriever,
    doc_title: str,
) -> list[NoteAnswer]:
    """Answer each question in parallel using BM25-selected note chunks."""
    if not questions:
        raise GenerationError("No questions provided. Paste or upload some questions.")

    qs = questions[:MAX_QUESTIONS]
    results: list[NoteAnswer | None] = [None] * len(qs)

    def work(i: int, q: str) -> tuple[int, NoteAnswer]:
        hits = retriever.search(q, top_k=TOP_K_CHUNKS) if chunks else []
        if hits:
            ctx = [_trim(chunks[idx]) for idx, _ in hits if idx < len(chunks)]
        else:
            # Even sample fallback
            if not chunks:
                ctx = []
            elif len(chunks) <= TOP_K_CHUNKS:
                ctx = [_trim(c) for c in chunks]
            else:
                step = len(chunks) / TOP_K_CHUNKS
                ctx = [
                    _trim(chunks[min(int(j * step), len(chunks) - 1)])
                    for j in range(TOP_K_CHUNKS)
                ]
        return i, _answer_one(question=q, chunks=ctx, doc_title=doc_title)

    workers = min(len(qs), 4)
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, i, q): i for i, q in enumerate(qs)}
        for fut in as_completed(futs):
            try:
                i, ans = fut.result()
                results[i] = ans
            except GenerationError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(str(e))

    answers = [a for a in results if a is not None]
    if not answers:
        raise GenerationError(
            errors[0] if errors else "Could not generate any answers. Try again."
        )
    return answers
