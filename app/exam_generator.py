"""Exam Quiz generation — Nigerian university theory-paper style.

Speed strategy:
  - Generate each major QUESTION in its own Claude call (small JSON).
  - Run those calls in parallel so wall-clock ≈ one question, not N.
  - Use a lean schema (no long guide_points on the critical path).
  - Cap / trim context chunks so prompts stay small and reliable.

Quality is preserved by assigning each question a different slice of the
document (syllabus coverage) rather than dumping the whole PDF into one call.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from .generator import GenerationError, MODEL, _normalize_difficulty

__all__ = [
    "ExamPaper",
    "ExamQuestion",
    "ExamPart",
    "ExamSubPart",
    "generate_exam_paper",
    "fill_answer_guides",
    "paper_total_marks",
    "GenerationError",
]

# Per-question context budget (speed + fewer 400s).
MAX_CHUNKS_PER_QUESTION = 6
MAX_WORDS_PER_CHUNK = 110
MAX_TOKENS_PER_QUESTION = 3500
ORDINALS = [
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
]


class ExamSubPart(BaseModel):
    label: str  # i, ii, iii
    text: str


class ExamPart(BaseModel):
    label: str  # a, b, c, …
    prompt: str
    marks: float
    subparts: list[ExamSubPart] = Field(default_factory=list)
    # Filled by fill_answer_guides after the student has attempted the paper.
    guide_points: list[str] = Field(default_factory=list)
    answer_outline: str = ""  # short model-answer sketch from the notes
    source_quote: str = ""


class ExamQuestion(BaseModel):
    number: int
    heading: str
    parts: list[ExamPart]


class ExamPaper(BaseModel):
    institution: str = "StudyQuiz Exam Prep"
    faculty: str = "Faculty of Science"
    department: str = "Department of Computer Science"
    exam_title: str = "BSc. Degree Examination (Practice Paper)"
    session: str = "Practice Session"
    course_code: str = "CSC XXX"
    course_title: str = "Course Title"
    course_unit: str = "2"
    time_allowed: str = "2 Hrs."
    instructions: list[str]
    questions: list[ExamQuestion]


# Lean schema for a single major question (critical path — no guide essays).
class _PartFast(BaseModel):
    label: str
    prompt: str
    marks: float
    subparts: list[ExamSubPart] = Field(default_factory=list)
    source_quote: str = ""


class _QuestionFast(BaseModel):
    number: int
    heading: str
    parts: list[_PartFast]


EXAM_SYSTEM_PROMPT = (
    "You are a chief examiner setting a Nigerian university undergraduate "
    "Computer Science written examination (BSc level), in the style used by "
    "institutions such as FUOYE and similar Nigerian universities.\n\n"
    "Rules:\n"
    "- Long-form theory ONLY (not multiple choice).\n"
    "- One major question only in this response (QUESTION N).\n"
    "- Parts (a)(b)(c)… with realistic marks (2–8 each).\n"
    "- Roman (i)(ii)(iii) only when listing several named items to discuss.\n"
    "- Verbs: Discuss, Define, Explain, Describe, List and explain, Compare…\n"
    "- Keep prompts concise (1–3 sentences per part).\n"
    "- If material is provided, every part must be answerable from it; "
    "source_quote = a short verbatim phrase (or empty if none).\n"
    "- Do not invent extra major questions."
)

EXAM_DIFFICULTY = {
    "easy": (
        "Difficulty EASY — Bloom: remember/understand. "
        "Prefer verbs: Define, List, State, Identify, Briefly explain. "
        "About 4–5 parts; marks mostly 2–5. Avoid evaluate/critique."
    ),
    "medium": (
        "Difficulty MEDIUM — Bloom: apply/analyse. "
        "Prefer verbs: Explain, Describe, Apply, Compare, Distinguish, Illustrate. "
        "About 4–6 parts; marks 3–6. Mix recall with application."
    ),
    "hard": (
        "Difficulty HARD — Bloom: evaluate/create. "
        "Prefer verbs: Evaluate, Critically examine, Justify, Design, Argue, "
        "Assess, Synthesise. About 5–6 parts; marks 4–8; use roman lists where useful. "
        "Require judgment or multi-step reasoning grounded in the material."
    ),
}


def _client():
    import anthropic

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise GenerationError(
            "No Anthropic API key configured. Set the ANTHROPIC_API_KEY "
            "environment variable before starting the server "
            "(get a key at https://platform.claude.com/)."
        )
    return anthropic.Anthropic()


def _api_error_message(exc: Exception) -> str:
    """Pull a useful message out of Anthropic errors (not just status code)."""
    parts = [str(getattr(exc, "message", "") or "")]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or body
        if isinstance(err, dict):
            parts.append(str(err.get("message") or err.get("type") or ""))
        else:
            parts.append(str(err))
    elif body:
        parts.append(str(body)[:400])
    text = " — ".join(p for p in parts if p).strip(" —")
    return text or type(exc).__name__


def _trim_chunk(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "…"


def assign_chunk_slices(
    chunks: list[str],
    num_questions: int,
    *,
    per_question: int = MAX_CHUNKS_PER_QUESTION,
) -> list[list[str]]:
    """Give each major question a different region of the document.

    Spreads coverage across long notes without sending the whole file once.
    """
    if num_questions <= 0:
        return []
    if not chunks:
        return [[] for _ in range(num_questions)]

    n = len(chunks)
    slices: list[list[str]] = []
    for q in range(num_questions):
        # Segment of the document for this question
        start = int(q * n / num_questions)
        end = int((q + 1) * n / num_questions)
        end = max(end, start + 1)
        segment = chunks[start:end]
        if len(segment) <= per_question:
            picked = segment
        else:
            # Even sample within the segment
            step = len(segment) / per_question
            picked = [segment[min(int(i * step), len(segment) - 1)] for i in range(per_question)]
        slices.append([_trim_chunk(c) for c in picked])
    return slices


def assign_topic_slices(
    chunks: list[str],
    num_questions: int,
    topic: str,
    retriever,
    *,
    per_question: int = MAX_CHUNKS_PER_QUESTION,
) -> list[list[str]]:
    """Retrieve note chunks relevant to ``topic`` and share them across questions.

    Uses BM25 so the paper stays on the focus topic instead of random pages.
    """
    if num_questions <= 0:
        return []
    if not chunks:
        return [[] for _ in range(num_questions)]

    topic = (topic or "").strip()
    if not topic or retriever is None:
        return assign_chunk_slices(chunks, num_questions, per_question=per_question)

    # Pull a large pool of topic-relevant chunks, then diversify.
    pool_k = min(len(chunks), max(per_question * num_questions * 2, 12))
    ranked = retriever.search(topic, top_k=pool_k)
    if not ranked:
        # Try a looser query if exact topic words are rare in the index
        ranked = retriever.search(topic.replace(",", " "), top_k=pool_k)
    if not ranked:
        return assign_chunk_slices(chunks, num_questions, per_question=per_question)

    try:
        from .coverage import diversify_indices

        diversified = diversify_indices(
            ranked, top_k=min(len(ranked), per_question * num_questions), n_chunks=len(chunks)
        )
        indices = [i for i, _ in diversified]
    except Exception:
        indices = [i for i, _ in ranked[: per_question * num_questions]]

    # Round-robin so each major question gets topic-relevant material
    slices: list[list[str]] = [[] for _ in range(num_questions)]
    for j, idx in enumerate(indices):
        if 0 <= idx < len(chunks):
            slices[j % num_questions].append(_trim_chunk(chunks[idx]))

    # Cap per question and ensure none empty
    for q in range(num_questions):
        if not slices[q]:
            # Fallback: top overall topic hits
            slices[q] = [
                _trim_chunk(chunks[i]) for i, _ in ranked[:per_question] if i < len(chunks)
            ]
        else:
            # Deduplicate while preserving order
            seen: set[str] = set()
            uniq: list[str] = []
            for c in slices[q]:
                if c in seen:
                    continue
                seen.add(c)
                uniq.append(c)
            slices[q] = uniq[:per_question]

    return slices


def _generate_one_question(
    *,
    question_number: int,
    num_questions: int,
    doc_title: str,
    context_chunks: list[str],
    topic: str | None,
    use_rag: bool,
    difficulty: str,
) -> ExamQuestion:
    """One Claude call → one major QUESTION (fast path)."""
    import anthropic

    diff = _normalize_difficulty(difficulty)
    ordinal = ORDINALS[question_number - 1] if question_number <= len(ORDINALS) else str(question_number)
    is_first = question_number == 1
    topic = (topic or "").strip()

    if use_rag and context_chunks:
        sources = "\n\n".join(
            f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        material = (
            f"Base this question STRICTLY on material from \"{doc_title}\":\n\n{sources}"
        )
    else:
        material = (
            f"No extract provided. Use standard syllabus knowledge for "
            f"\"{doc_title}\". Leave source_quote empty on every part."
        )

    if topic:
        role = (
            "This is COMPULSORY QUESTION ONE — slightly broader within the focus topic, more parts."
            if is_first
            else f"This is an optional question (QUESTION {ordinal}) — another angle on the SAME focus topic."
        )
        topic_block = f"""
FOCUS TOPIC:
"{topic}"

Every part of this question must examine an aspect of this focus topic.
- Do not set questions on unrelated chapters of the course.
- Sub-parts, definitions, and discussions stay inside this topic.
- You may cover different sub-themes of the topic across parts (e.g. definition, types, advantages, comparison, application) — still all under "{topic}".
"""
    else:
        role = (
            "This is COMPULSORY QUESTION ONE — slightly broader, more parts."
            if is_first
            else f"This is an optional question (QUESTION {ordinal}) — a different angle from the notes."
        )
        topic_block = ""
    part_count = "5–6" if is_first else "3–5"

    task = f"""Set exactly ONE major theory question for a BSc practice paper.

{role}
- number: {question_number}
- heading: "QUESTION {ordinal}"
- About {part_count} lettered parts with marks.
- Use roman sub-parts only when listing several items to discuss.
{EXAM_DIFFICULTY[diff]}
{topic_block}
{material}

Respond with structured fields only for this single question.
"""

    client = _client()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS_PER_QUESTION,
            system=EXAM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task}],
            output_format=_QuestionFast,
        )
    except anthropic.AuthenticationError:
        raise GenerationError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise GenerationError("Rate limited by the Claude API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        detail = _api_error_message(e)
        raise GenerationError(
            f"Claude API error ({e.status_code}): {detail}"
        ) from e
    except anthropic.APIConnectionError:
        raise GenerationError("Could not reach the Claude API. Check your internet connection.")
    except Exception as e:
        raise GenerationError(f"Exam generation failed: {_api_error_message(e)}") from e

    raw = response.parsed_output
    if raw is None or not raw.parts:
        raise GenerationError(
            f"QUESTION {ordinal} came back empty. Try again with fewer questions."
        )

    parts: list[ExamPart] = []
    for p in raw.parts:
        if not (p.prompt or "").strip() or p.marks <= 0:
            continue
        parts.append(
            ExamPart(
                label=(p.label or "a").strip()[:3],
                prompt=p.prompt.strip(),
                marks=float(p.marks),
                subparts=list(p.subparts or []),
                guide_points=[],  # deferred — keeps this call small/fast
                source_quote=(p.source_quote or "").strip(),
            )
        )
    if not parts:
        raise GenerationError(f"QUESTION {ordinal} had no valid parts. Try again.")

    heading = (raw.heading or f"QUESTION {ordinal}").strip().upper()
    if not heading.startswith("QUESTION"):
        heading = f"QUESTION {ordinal}"

    return ExamQuestion(number=question_number, heading=heading, parts=parts)


def _run_parallel_questions(
    *,
    indices: list[int],
    num_questions: int,
    doc_title: str,
    slices: list[list[str]],
    topic: str | None,
    use_rag: bool,
    difficulty: str,
    questions: list[ExamQuestion | None],
    errors: list[str],
) -> None:
    """Fill ``questions[i]`` for each i in ``indices`` (parallel). Mutates in place."""
    if not indices:
        return
    workers = min(len(indices), 4)
    with ThreadPoolExecutor(max_workers=workers) as pool_exec:
        futures = {
            pool_exec.submit(
                _generate_one_question,
                question_number=i + 1,
                num_questions=num_questions,
                doc_title=doc_title,
                context_chunks=slices[i] if i < len(slices) else [],
                topic=topic,
                use_rag=use_rag,
                difficulty=difficulty,
            ): i
            for i in indices
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                questions[idx] = fut.result()
            except GenerationError as e:
                errors.append(f"QUESTION {idx + 1}: {e}")
            except Exception as e:
                errors.append(f"QUESTION {idx + 1} failed: {e}")


def build_exam_generation_meta(
    *,
    requested: int,
    generated: int,
    failed_slots: list[int],
    retried_slots: list[int],
    errors: list[str],
) -> dict:
    """Public meta for API/UI (partial papers no longer silent)."""
    partial = generated < requested
    missing = max(0, requested - generated)
    if not partial:
        message = None
    elif generated == 0:
        message = "No major questions could be generated. Try again."
    else:
        failed_labels = ", ".join(
            f"QUESTION {ORDINALS[s - 1] if 1 <= s <= len(ORDINALS) else s}"
            for s in failed_slots
        ) or "some slots"
        retry_note = (
            " after one automatic retry"
            if retried_slots
            else ""
        )
        message = (
            f"{generated} of {requested} major questions generated"
            f" ({failed_labels} failed{retry_note}). "
            "Generate again to fill the missing slot(s)."
        )
    return {
        "requested": requested,
        "generated": generated,
        "missing_count": missing,
        "partial": partial,
        "failed_slots": list(failed_slots),
        "retried_slots": list(retried_slots),
        "errors": list(errors),
        "message": message,
    }


def renumber_exam_questions(questions: list[ExamQuestion]) -> list[ExamQuestion]:
    """Assign consecutive QUESTION ONE… headings after drops/retries."""
    for i, q in enumerate(questions):
        ordinal = ORDINALS[i] if i < len(ORDINALS) else str(i + 1)
        q.number = i + 1
        q.heading = f"QUESTION {ordinal}"
    return questions


def generate_exam_paper(
    *,
    num_questions: int,
    doc_title: str,
    context_chunks: list[str],
    topic: str | None,
    use_rag: bool,
    difficulty: str = "medium",
    course_code: str | None = None,
    course_title: str | None = None,
    time_allowed: str = "2 Hrs.",
    all_document_chunks: list[str] | None = None,
    retriever=None,
) -> tuple[ExamPaper, dict]:
    """Build a full paper by generating major questions in parallel.

    Returns ``(paper, generation_meta)``. Failed slots are retried once; if
    still incomplete, the partial paper is returned with an honest meta
    message (e.g. "3 of 4 generated") instead of silently shrinking.

    ``all_document_chunks`` (preferred) is sliced across questions for coverage.
    When ``topic`` + ``retriever`` are set, chunks are BM25-selected for that topic.
    """
    num_questions = max(2, min(num_questions, 6))
    code = (course_code or "").strip() or _guess_code(doc_title)
    title = (course_title or "").strip() or _guess_title(doc_title)

    pool = all_document_chunks if all_document_chunks is not None else context_chunks
    topic_clean = (topic or "").strip()
    if use_rag:
        # When a focus topic is set, retrieve topic-relevant chunks (not random spread).
        if topic_clean and retriever is not None:
            slices = assign_topic_slices(
                pool or [],
                num_questions,
                topic_clean,
                retriever,
            )
        else:
            slices = assign_chunk_slices(pool or [], num_questions)
    else:
        slices = [[] for _ in range(num_questions)]

    questions: list[ExamQuestion | None] = [None] * num_questions
    errors: list[str] = []
    retried_slots: list[int] = []

    # Pass 1 — all slots in parallel (wall time ≈ slowest single question).
    _run_parallel_questions(
        indices=list(range(num_questions)),
        num_questions=num_questions,
        doc_title=doc_title,
        slices=slices,
        topic=topic_clean or None,
        use_rag=use_rag,
        difficulty=difficulty,
        questions=questions,
        errors=errors,
    )

    # Pass 2 — retry only failed slots once.
    failed_first = [i for i, q in enumerate(questions) if q is None]
    if failed_first:
        retried_slots = [i + 1 for i in failed_first]
        # Drop stale errors for slots we are about to retry; keep others if any.
        retry_errors: list[str] = []
        _run_parallel_questions(
            indices=failed_first,
            num_questions=num_questions,
            doc_title=doc_title,
            slices=slices,
            topic=topic_clean or None,
            use_rag=use_rag,
            difficulty=difficulty,
            questions=questions,
            errors=retry_errors,
        )
        errors.extend(retry_errors)

    failed_final = [i + 1 for i, q in enumerate(questions) if q is None]
    cleaned = [q for q in questions if q is not None]
    if not cleaned:
        raise GenerationError(
            errors[0] if errors else "No exam questions were generated. Try again."
        )

    # Renumber only after retries so headings match the paper the student sees.
    renumber_exam_questions(cleaned)

    instructions = [
        f"Answer Question One and Any Other {max(1, len(cleaned) - 1)} Question(s).",
        "No jotting is allowed on the Question Paper.",
        "Credit will be given for clarity, structure, and relevant examples.",
    ]

    paper = ExamPaper(
        course_code=code,
        course_title=title,
        time_allowed=time_allowed or "2 Hrs.",
        instructions=instructions,
        questions=cleaned,
    )
    meta = build_exam_generation_meta(
        requested=num_questions,
        generated=len(cleaned),
        failed_slots=failed_final,
        retried_slots=retried_slots,
        errors=errors,
    )
    return paper, meta


def _guess_code(doc_title: str) -> str:
    import re

    m = re.search(r"\b([A-Z]{2,4}\s*\d{2,3})\b", doc_title or "", re.I)
    if m:
        return m.group(1).upper().replace("  ", " ")
    return "CSC 000"


def _guess_title(doc_title: str) -> str:
    name = (doc_title or "Course Materials").rsplit(".", 1)[0]
    name = name.replace("_", " ").replace("-", " ").strip()
    return name[:80] if name else "Course Materials"


def paper_total_marks(paper: ExamPaper) -> float:
    return sum(p.marks for q in paper.questions for p in q.parts)


# ---------------------------------------------------------------------------
# Answer / marking guides (run AFTER the student has practised on paper)
# ---------------------------------------------------------------------------


class _PartAnswer(BaseModel):
    label: str
    guide_points: list[str] = Field(default_factory=list)
    answer_outline: str = ""
    source_quote: str = ""


class _QuestionAnswers(BaseModel):
    parts: list[_PartAnswer]


GUIDE_SYSTEM = (
    "You are a university examiner preparing a marking scheme and model-answer "
    "outline for a BSc theory paper. Base every point STRICTLY on the provided "
    "lecture material. Do not invent facts that are not supported by the sources. "
    "Write for a Nigerian undergraduate student revising after attempting the paper. "
    "answer_outline: 4–8 sentences a strong student answer should cover. "
    "guide_points: 3–6 short bullets (marking scheme style). "
    "source_quote: short verbatim phrase from the material (or empty)."
)


def _context_for_question(
    question: ExamQuestion,
    chunks: list[str],
    retriever,
) -> list[str]:
    """Retrieve the most relevant note chunks for this major question."""
    if not chunks:
        return []
    # Query = question heading + all part prompts
    bits = [question.heading or ""]
    for p in question.parts:
        bits.append(p.prompt)
        for sp in p.subparts or []:
            bits.append(sp.text)
    query = " ".join(bits)
    if retriever is not None:
        hits = retriever.search(query, top_k=MAX_CHUNKS_PER_QUESTION)
        if hits:
            return [_trim_chunk(chunks[i]) for i, _ in hits if i < len(chunks)]
    # Fallback: even sample
    if len(chunks) <= MAX_CHUNKS_PER_QUESTION:
        return [_trim_chunk(c) for c in chunks]
    step = len(chunks) / MAX_CHUNKS_PER_QUESTION
    return [
        _trim_chunk(chunks[min(int(i * step), len(chunks) - 1)])
        for i in range(MAX_CHUNKS_PER_QUESTION)
    ]


def _generate_answers_for_question(
    *,
    question: ExamQuestion,
    doc_title: str,
    context_chunks: list[str],
) -> dict[str, _PartAnswer]:
    """One Claude call → marking points + outline for every part of one QUESTION."""
    import anthropic

    parts_desc = []
    for p in question.parts:
        sub = ""
        if p.subparts:
            sub = " Sub-parts: " + "; ".join(f"{s.label}. {s.text}" for s in p.subparts)
        parts_desc.append(
            f"({p.label}) [{p.marks} marks] {p.prompt}{sub}"
        )

    if context_chunks:
        sources = "\n\n".join(
            f"[Source {i + 1}]\n{c}" for i, c in enumerate(context_chunks)
        )
        material = f"Lecture material from \"{doc_title}\":\n\n{sources}"
    else:
        material = (
            f"No lecture extract. Use standard knowledge for \"{doc_title}\" "
            f"and leave source_quote empty."
        )

    task = f"""Produce a marking scheme for this exam question ONLY.

{question.heading} (question number {question.number})

Parts:
{chr(10).join(parts_desc)}

For EACH part, return:
- label (same letter)
- guide_points: 3–6 bullets a marker would tick
- answer_outline: a concise model answer (not a full essay; enough to self-check)
- source_quote: short support from the material

{material}
"""

    client = _client()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS_PER_QUESTION,
            system=GUIDE_SYSTEM,
            messages=[{"role": "user", "content": task}],
            output_format=_QuestionAnswers,
        )
    except anthropic.AuthenticationError:
        raise GenerationError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise GenerationError("Rate limited by the Claude API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        raise GenerationError(
            f"Claude API error ({e.status_code}): {_api_error_message(e)}"
        ) from e
    except anthropic.APIConnectionError:
        raise GenerationError("Could not reach the Claude API. Check your internet connection.")
    except Exception as e:
        raise GenerationError(f"Answer guide generation failed: {_api_error_message(e)}") from e

    raw = response.parsed_output
    if raw is None or not raw.parts:
        raise GenerationError(
            f"No answers returned for {question.heading}. Try again."
        )

    by_label: dict[str, _PartAnswer] = {}
    for p in raw.parts:
        lab = (p.label or "").strip().lower()[:3]
        if not lab:
            continue
        by_label[lab] = p
    return by_label


def fill_answer_guides(
    paper: ExamPaper,
    *,
    doc_title: str,
    chunks: list[str],
    retriever=None,
) -> ExamPaper:
    """Attach model-answer outlines + marking points to an existing paper.

    Safe to call after the student has written their own answers. Runs one
    Claude call per major question, in parallel, each grounded in BM25-retrieved
    notes for that question.
    """
    if not paper.questions:
        return paper

    def work(q: ExamQuestion) -> ExamQuestion:
        ctx = _context_for_question(q, chunks, retriever)
        answers = _generate_answers_for_question(
            question=q,
            doc_title=doc_title,
            context_chunks=ctx,
        )
        new_parts: list[ExamPart] = []
        for p in q.parts:
            key = (p.label or "").strip().lower()[:3]
            a = answers.get(key)
            if a is None:
                # try first letter only
                a = answers.get(key[:1]) if key else None
            if a is None:
                new_parts.append(p)
                continue
            new_parts.append(
                ExamPart(
                    label=p.label,
                    prompt=p.prompt,
                    marks=p.marks,
                    subparts=p.subparts,
                    guide_points=list(a.guide_points or [])[:8],
                    answer_outline=(a.answer_outline or "").strip(),
                    source_quote=(a.source_quote or p.source_quote or "").strip(),
                )
            )
        return ExamQuestion(number=q.number, heading=q.heading, parts=new_parts)

    updated: list[ExamQuestion | None] = [None] * len(paper.questions)
    errors: list[str] = []
    workers = min(len(paper.questions), 4)
    with ThreadPoolExecutor(max_workers=workers) as pool_exec:
        futures = {
            pool_exec.submit(work, q): i for i, q in enumerate(paper.questions)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                updated[i] = fut.result()
            except GenerationError as e:
                errors.append(str(e))
                updated[i] = paper.questions[i]
            except Exception as e:
                errors.append(str(e))
                updated[i] = paper.questions[i]

    paper.questions = [q for q in updated if q is not None]
    if not any(
        (p.guide_points or p.answer_outline)
        for q in paper.questions
        for p in q.parts
    ):
        raise GenerationError(
            errors[0]
            if errors
            else "Could not generate answer guides. Try again."
        )
    return paper
