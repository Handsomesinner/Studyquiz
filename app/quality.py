"""Quiz quality helpers: near-duplicate filtering and grounding retry selection."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import QuizQuestion
    from .grounding import QuestionGrounding


_TOKEN = re.compile(r"[a-z0-9]+")


def normalize_stem(text: str) -> str:
    return " ".join(_TOKEN.findall((text or "").casefold()))


def jaccard(a: str, b: str) -> float:
    ta = set(_TOKEN.findall((a or "").casefold()))
    tb = set(_TOKEN.findall((b or "").casefold()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def options_jaccard(opts_a: list[str], opts_b: list[str]) -> float:
    """Max pairwise option similarity (detect copy-paste option sets)."""
    if not opts_a or not opts_b:
        return 0.0
    best = 0.0
    for x in opts_a:
        for y in opts_b:
            best = max(best, jaccard(x, y))
    # Also compare as bags of all option tokens
    bag = jaccard(" ".join(opts_a), " ".join(opts_b))
    return max(best, bag)


def is_near_duplicate(
    q_a: "QuizQuestion",
    q_b: "QuizQuestion",
    *,
    stem_threshold: float = 0.72,
    options_threshold: float = 0.85,
) -> bool:
    stem_sim = jaccard(q_a.question, q_b.question)
    if stem_sim >= stem_threshold:
        return True
    # Same stem after aggressive normalize
    if normalize_stem(q_a.question) == normalize_stem(q_b.question):
        return True
    if options_jaccard(list(q_a.options), list(q_b.options)) >= options_threshold:
        # Only flag option-set clones when stems are at least somewhat related
        if stem_sim >= 0.35:
            return True
    return False


def filter_near_duplicates(
    questions: list["QuizQuestion"],
    groundings: list["QuestionGrounding"] | None = None,
    *,
    stem_threshold: float = 0.72,
) -> tuple[list["QuizQuestion"], list["QuestionGrounding"] | None, int]:
    """Keep first occurrence; drop later near-duplicates. Returns (qs, gs, dropped)."""
    kept_q: list = []
    kept_g: list = []
    dropped = 0
    has_g = groundings is not None and len(groundings) == len(questions)
    for i, q in enumerate(questions):
        if any(
            is_near_duplicate(q, prev, stem_threshold=stem_threshold) for prev in kept_q
        ):
            dropped += 1
            continue
        kept_q.append(q)
        if has_g:
            kept_g.append(groundings[i])  # type: ignore[index]
    if not has_g:
        return kept_q, None, dropped
    return kept_q, kept_g, dropped


def ungrounded_indices(
    groundings: list["QuestionGrounding"],
    *,
    use_rag: bool,
) -> list[int]:
    """Indices of questions that failed grounding when RAG expects quotes."""
    if not use_rag:
        return []
    return [i for i, g in enumerate(groundings) if not g.grounded]
