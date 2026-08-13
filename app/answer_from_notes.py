"""Answer user-supplied questions using retrieved lecture notes (RAG).

Students paste or upload their own questions (e.g. past papers). For each
question we BM25-retrieve note chunks and ask Claude for a short outline plus
a fuller model answer grounded only in those excerpts.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from pydantic import BaseModel, Field

from .generator import MODEL, GenerationError, _client
from .retriever import BM25Retriever

MAX_QUESTIONS = 20
TOP_K_CHUNKS = 6
MAX_WORDS_PER_CHUNK = 140

# Numbered: 1.  1)  1:  Q1.  Question 1:
_NUM_START = re.compile(
    r"(?m)^\s*(?:Q(?:uestion)?\s*)?(\d{1,3})[\.\)\:]\s+",
    re.IGNORECASE,
)
_QUESTION_N = re.compile(
    r"(?m)^\s*Question\s+(\d{1,3})\s*[:.\-–—]\s*",
    re.IGNORECASE,
)
# Lettered parts often used as separate items: a) b) (a) (b) i. ii.
_LETTER_START = re.compile(
    r"(?m)^\s*[\(\[]?([a-z]|[ivx]{1,4})[\)\].:]\s+",
    re.IGNORECASE,
)


class NoteAnswer(BaseModel):
    question: str
    outline: List[str] = Field(default_factory=list)
    full_answer: str = ""
    source_quotes: List[str] = Field(default_factory=list)
    notes_cover_question: bool = True
    error: Optional[str] = None  # set when this question failed to generate


class _AnswerOut(BaseModel):
    outline: List[str] = Field(default_factory=list)
    full_answer: str = ""
    source_quotes: List[str] = Field(default_factory=list)
    notes_cover_question: bool = True


def parse_questions(text: str) -> List[str]:
    """Split free-form past-question text into individual questions.

    Supports:
    - 1. / 1) / Q1. / Question 1:
    - a) / (b) / i. lettered or roman lines (when no numbers)
    - blank-line separated blocks
    - plain one-question-per-line lists
    """
    if not text or not text.strip():
        return []

    raw = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    raw = re.sub(r"[•●▪]", "-", raw)

    parts = _split_by_pattern(raw, _NUM_START)
    if len(parts) < 2:
        parts = _split_by_pattern(raw, _QUESTION_N)
    if len(parts) < 2:
        # Only use lettered split if we see several lettered starts
        letter_hits = list(_LETTER_START.finditer(raw))
        if len(letter_hits) >= 2:
            parts = _split_by_pattern(raw, _LETTER_START)
    if len(parts) < 2:
        blocks = re.split(r"\n\s*\n+", raw)
        if len(blocks) > 1:
            parts = [_clean_question(b) for b in blocks]
        else:
            parts = [_clean_question(line) for line in raw.split("\n")]

    # Deduplicate (case-insensitive), drop empties
    seen: set[str] = set()
    out: list[str] = []
    for q in parts:
        q = _clean_question(q)
        if not q:
            continue
        key = re.sub(r"\s+", " ", q.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _split_by_pattern(raw: str, pattern: re.Pattern) -> list[str]:
    matches = list(pattern.finditer(raw))
    if not matches:
        return []
    parts: list[str] = []
    # Text before first marker (if long enough) as its own question
    head = raw[: matches[0].start()].strip()
    if head and len(head) > 12:
        parts.append(head)
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if body:
            parts.append(body)
    return parts


def _clean_question(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
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

    task = f"""Answer this student exam question using ONLY the lecture excerpts below.

Document: "{doc_title}"

Question:
{question}

Lecture excerpts:
{sources}

Write for a student revising for a university exam who needs to understand fast.

full_answer (MAIN answer — shown first):
- Clear exam-style model answer a student can study from.
- Use short paragraphs and simple language.
- Where helpful, start sections with plain labels ending in a colon, e.g.
  Functionality: ... or Definition: ...
  Do NOT use markdown asterisks like **bold** — the app will style labels itself.
- Cover every part of a multi-part question in order.
- 2–5 short paragraphs (or short labelled sections) — not a long essay.
- Easy to skim and remember under exam pressure.

outline (SUMMARY — short recap only):
- 3–6 very short bullets for quick revision after reading the full answer.
- Keywords and phrases only (not full sentences if a phrase is enough).

Other rules:
- If excerpts are insufficient, set notes_cover_question=false and say what is missing.
- source_quotes: 0–3 short verbatim phrases from the excerpts (empty if none).
- Do not invent content that is not supported by the excerpts.
"""

    client = _client()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=4500,
            system=(
                "You are a university exam tutor. Answer only from provided lecture notes. "
                "Write so a student can understand and revise quickly: clear structure, "
                "exam wording, no fluff."
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
        error=None,
    )


def answer_questions(
    *,
    questions: list[str],
    chunks: list[str],
    retriever: BM25Retriever,
    doc_title: str,
) -> list[NoteAnswer]:
    """Answer each question in parallel using BM25-selected note chunks.

    Always returns one entry per input question (failed ones include ``error``)
    so the UI never silently drops items.
    """
    if not questions:
        raise GenerationError("No questions provided. Paste or upload some questions.")

    qs = questions[:MAX_QUESTIONS]
    results: list[NoteAnswer | None] = [None] * len(qs)

    def work(i: int, q: str) -> tuple[int, NoteAnswer]:
        hits = retriever.search(q, top_k=TOP_K_CHUNKS) if chunks else []
        if hits:
            ctx = [_trim(chunks[idx]) for idx, _ in hits if idx < len(chunks)]
        else:
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
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, i, q): i for i, q in enumerate(qs)}
        for fut in as_completed(futs):
            i = futs[fut]
            q = qs[i]
            try:
                idx, ans = fut.result()
                results[idx] = ans
            except GenerationError as e:
                results[i] = NoteAnswer(
                    question=q,
                    outline=[],
                    full_answer="",
                    source_quotes=[],
                    notes_cover_question=False,
                    error=str(e),
                )
            except Exception as e:
                results[i] = NoteAnswer(
                    question=q,
                    outline=[],
                    full_answer="",
                    source_quotes=[],
                    notes_cover_question=False,
                    error=str(e),
                )

    answers = [
        r
        if r is not None
        else NoteAnswer(
            question=qs[i],
            outline=[],
            full_answer="",
            notes_cover_question=False,
            error="No answer was generated for this question.",
        )
        for i, r in enumerate(results)
    ]

    # If every single one failed with the same generation config issue, surface it
    if all(a.error for a in answers):
        raise GenerationError(answers[0].error or "Could not generate any answers.")

    return answers
