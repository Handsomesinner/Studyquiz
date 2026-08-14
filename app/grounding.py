"""Grounding validation: check that model-claimed source quotes appear in the document.

This is the automatic side of the evaluation chapter. RAG questions must cite a
short verbatim quote from the lecture material; we verify that claim rather than
trusting the model. The resulting quote-in-source rate is a hard metric that
compares RAG vs the ungrounded baseline (where quotes are empty by design).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from .generator import QuizQuestion

# Quotes shorter than this (after normalising) are too vague to count as evidence.
MIN_QUOTE_CHARS = 12


def normalize_for_match(text: str) -> str:
    """Casefold, unify quotes/dashes, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    # Curly quotes / dashes → ASCII so PDF extraction quirks don't false-fail.
    table = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
        }
    )
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_loose(text: str) -> str:
    """Like normalize_for_match but strips punctuation (for soft matching)."""
    text = normalize_for_match(text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def quote_in_source(quote: str, source_text: str) -> tuple[bool, str]:
    """Return (found, match_type).

    match_type is one of:
      empty_quote | too_short | exact | normalized | not_found
    """
    if quote is None or not str(quote).strip():
        return False, "empty_quote"

    q_strict = normalize_for_match(quote)
    if len(q_strict) < MIN_QUOTE_CHARS:
        return False, "too_short"

    corpus_strict = normalize_for_match(source_text)
    if q_strict in corpus_strict:
        return True, "exact"

    q_loose = normalize_loose(quote)
    corpus_loose = normalize_loose(source_text)
    if len(q_loose) >= MIN_QUOTE_CHARS and q_loose in corpus_loose:
        return True, "normalized"

    return False, "not_found"


def options_are_unique(options: list[str]) -> bool:
    """True when all four options are distinct after normalisation."""
    norms = [normalize_for_match(o) for o in options]
    return len(norms) == len(set(norms)) and all(norms)


@dataclass
class QuestionGrounding:
    grounded: bool
    match_type: str
    source_quote: str
    options_unique: bool
    # True when this question is expected to be grounded (RAG mode).
    expected_grounded: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QuizGroundingMetrics:
    use_rag: bool
    total_questions: int
    grounded_count: int
    ungrounded_count: int
    quote_in_source_rate: float  # grounded_count / total, 0 if empty
    options_unique_rate: float
    match_type_counts: dict[str, int]
    # For baseline (use_rag=False): fraction of questions with empty quotes
    # (the intended behaviour). High is good for the baseline condition.
    empty_quote_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_question(
    question: QuizQuestion,
    *,
    source_text: str,
    use_rag: bool,
) -> QuestionGrounding:
    """Check one question's source_quote against the full document text."""
    found, match_type = quote_in_source(question.source_quote, source_text)
    unique = options_are_unique(question.options)

    if not use_rag:
        # Baseline: quotes should be empty; "grounded" is always False by design.
        empty = match_type == "empty_quote"
        return QuestionGrounding(
            grounded=False,
            match_type=match_type if not empty else "baseline_empty",
            source_quote=question.source_quote or "",
            options_unique=unique,
            expected_grounded=False,
        )

    return QuestionGrounding(
        grounded=found,
        match_type=match_type,
        source_quote=question.source_quote or "",
        options_unique=unique,
        expected_grounded=True,
    )


def validate_quiz(
    questions: list[QuizQuestion],
    *,
    source_text: str,
    use_rag: bool,
) -> tuple[list[QuestionGrounding], QuizGroundingMetrics]:
    results = [
        validate_question(q, source_text=source_text, use_rag=use_rag)
        for q in questions
    ]
    metrics = summarise(results, use_rag=use_rag)
    return results, metrics


def summarise(
    results: list[QuestionGrounding],
    *,
    use_rag: bool,
) -> QuizGroundingMetrics:
    total = len(results)
    grounded = sum(1 for r in results if r.grounded)
    unique = sum(1 for r in results if r.options_unique)
    empty = sum(
        1
        for r in results
        if r.match_type in ("empty_quote", "baseline_empty")
    )
    match_counts: dict[str, int] = {}
    for r in results:
        match_counts[r.match_type] = match_counts.get(r.match_type, 0) + 1

    return QuizGroundingMetrics(
        use_rag=use_rag,
        total_questions=total,
        grounded_count=grounded,
        ungrounded_count=total - grounded,
        quote_in_source_rate=(grounded / total) if total else 0.0,
        options_unique_rate=(unique / total) if total else 0.0,
        match_type_counts=match_counts,
        empty_quote_rate=(empty / total) if total else 0.0,
    )


def filter_grounded(
    questions: list[QuizQuestion],
    groundings: list[QuestionGrounding],
) -> tuple[list[QuizQuestion], list[QuestionGrounding]]:
    """Keep only RAG questions whose source quote was found in the document."""
    kept_q: list[QuizQuestion] = []
    kept_g: list[QuestionGrounding] = []
    for q, g in zip(questions, groundings):
        if g.grounded:
            kept_q.append(q)
            kept_g.append(g)
    return kept_q, kept_g


def resolve_grounded_serving(
    questions: list[QuizQuestion],
    groundings: list[QuestionGrounding],
    *,
    use_rag: bool,
    require_grounding: bool,
) -> tuple[list[QuizQuestion], list[QuestionGrounding], int, bool]:
    """Apply require_grounding; never return empty when questions exist.

    Returns ``(served_questions, served_groundings, filtered_out, best_effort)``.

    When strict mode would drop every question, falls back to the full model
    output with ``best_effort=True`` so the student still gets a quiz.
    """
    if not questions:
        return [], [], 0, False
    if not (use_rag and require_grounding):
        return list(questions), list(groundings), 0, False

    kept_q, kept_g = filter_grounded(questions, groundings)
    filtered_out = len(questions) - len(kept_q)
    if kept_q:
        return kept_q, kept_g, filtered_out, False
    return list(questions), list(groundings), 0, True


# ---------------------------------------------------------------------------
# Soft badges for exam parts & My-questions answers (never drop content)
# ---------------------------------------------------------------------------

# status: verified | partial | not_found
_SOFT_LABELS = {
    "verified": "Verified",
    "partial": "Partial",
    "not_found": "Not found",
}


def soft_badge_for_quote(quote: str | None, source_text: str) -> dict:
    """Classify one claimed quote for UI badges (does not filter content).

    - **verified** — quote found in the notes (exact or normalised)
    - **partial** — empty quote, or too short to count as solid evidence
    - **not_found** — quote claimed but not located in the notes
    """
    found, match_type = quote_in_source(quote or "", source_text)
    quote_s = (quote or "").strip()

    if match_type == "empty_quote":
        status = "partial"
        label = "No quote"
    elif match_type == "too_short":
        status = "partial"
        label = "Partial"
    elif found:
        status = "verified"
        label = "Verified"
    else:
        status = "not_found"
        label = "Not found"

    return {
        "status": status,
        "label": label,
        "match_type": match_type,
        "grounded": bool(found),
        "source_quote": quote_s,
    }


def aggregate_soft_badges(badges: list[dict]) -> dict:
    """Roll up per-quote badges into one overall badge for a multi-quote answer."""
    if not badges:
        return soft_badge_for_quote("", "")

    statuses = [b.get("status") or "not_found" for b in badges]
    verified_n = sum(1 for s in statuses if s == "verified")
    partial_n = sum(1 for s in statuses if s == "partial")
    not_found_n = sum(1 for s in statuses if s == "not_found")
    total = len(statuses)

    if verified_n == total:
        status = "verified"
        label = "Verified"
    elif verified_n > 0 and (partial_n > 0 or not_found_n > 0):
        status = "partial"
        label = "Partial"
    elif verified_n > 0:
        status = "verified"
        label = "Verified"
    elif not_found_n == total:
        status = "not_found"
        label = "Not found"
    else:
        # all partial / mix of partial only
        status = "partial"
        # Prefer "No quote" only when every badge was empty
        if all((b.get("match_type") == "empty_quote") for b in badges):
            label = "No quote"
        else:
            label = "Partial"

    return {
        "status": status,
        "label": label,
        "match_type": "aggregate",
        "grounded": verified_n == total and total > 0,
        "verified_count": verified_n,
        "partial_count": partial_n,
        "not_found_count": not_found_n,
        "total_quotes": total,
        "source_quote": "",
    }


def annotate_exam_paper_dict(paper: dict, source_text: str) -> dict:
    """Attach soft ``grounding`` badges to every exam part (in place). Never drops parts."""
    verified = partial = not_found = total = 0
    for q in paper.get("questions") or []:
        for p in q.get("parts") or []:
            total += 1
            g = soft_badge_for_quote(p.get("source_quote"), source_text)
            p["grounding"] = g
            st = g["status"]
            if st == "verified":
                verified += 1
            elif st == "partial":
                partial += 1
            else:
                not_found += 1
    paper["grounding_summary"] = {
        "total_parts": total,
        "verified": verified,
        "partial": partial,
        "not_found": not_found,
        "quote_in_source_rate": (verified / total) if total else 0.0,
        "mode": "soft_badges",
    }
    return paper


def annotate_note_answer_dict(answer: dict, source_text: str) -> dict:
    """Attach per-quote and overall soft badges on a My-questions answer dict."""
    quotes = list(answer.get("source_quotes") or [])
    quote_groundings = [soft_badge_for_quote(q, source_text) for q in quotes]
    if quote_groundings:
        overall = aggregate_soft_badges(quote_groundings)
    else:
        overall = soft_badge_for_quote("", source_text)
        overall["label"] = "No quote"
    answer["quote_groundings"] = quote_groundings
    answer["grounding"] = overall
    return answer


def annotate_note_answers(
    answers: list[dict],
    source_text: str,
) -> tuple[list[dict], dict]:
    """Annotate a list of answer dicts; return (answers, summary)."""
    verified = partial = not_found = 0
    out: list[dict] = []
    for a in answers:
        if a.get("error"):
            out.append(a)
            continue
        annotated = annotate_note_answer_dict(a, source_text)
        out.append(annotated)
        st = (annotated.get("grounding") or {}).get("status")
        if st == "verified":
            verified += 1
        elif st == "not_found":
            not_found += 1
        else:
            partial += 1
    total = verified + partial + not_found
    summary = {
        "total_answers": total,
        "verified": verified,
        "partial": partial,
        "not_found": not_found,
        "quote_in_source_rate": (verified / total) if total else 0.0,
        "mode": "soft_badges",
    }
    return out, summary
