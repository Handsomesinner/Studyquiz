"""Unit tests for source-quote grounding validation (no API key required)."""

from app.generator import QuizQuestion
from app.grounding import (
    filter_grounded,
    normalize_for_match,
    options_are_unique,
    quote_in_source,
    validate_question,
    validate_quiz,
)


def _q(quote: str, options=None) -> QuizQuestion:
    return QuizQuestion(
        question="What is X?",
        options=options or ["A", "B", "C", "D"],
        correct_index=0,
        explanation="Because the notes say so.",
        source_quote=quote,
    )


SOURCE = (
    "Process scheduling decides which ready process runs next. "
    "The round-robin algorithm uses a fixed time quantum. "
    "Priority scheduling can cause starvation of low-priority jobs."
)


def test_normalize_collapses_whitespace_and_case():
    assert normalize_for_match("  Hello   WORLD  ") == "hello world"


def test_quote_exact_match():
    found, kind = quote_in_source(
        "The round-robin algorithm uses a fixed time quantum.",
        SOURCE,
    )
    assert found is True
    assert kind == "exact"


def test_quote_normalized_punctuation():
    # Curly quotes / extra spaces should still match.
    found, kind = quote_in_source(
        "round-robin algorithm uses a fixed time quantum",
        SOURCE,
    )
    assert found is True
    assert kind in ("exact", "normalized")


def test_quote_not_found():
    found, kind = quote_in_source(
        "Virtual memory uses demand paging with thrashing thresholds.",
        SOURCE,
    )
    assert found is False
    assert kind == "not_found"


def test_empty_and_short_quotes():
    assert quote_in_source("", SOURCE) == (False, "empty_quote")
    assert quote_in_source("  ", SOURCE) == (False, "empty_quote")
    assert quote_in_source("hi", SOURCE)[1] == "too_short"


def test_options_unique():
    assert options_are_unique(["A", "B", "C", "D"]) is True
    assert options_are_unique(["A", "A", "C", "D"]) is False


def test_validate_rag_grounded():
    q = _q("Priority scheduling can cause starvation of low-priority jobs.")
    g = validate_question(q, source_text=SOURCE, use_rag=True)
    assert g.grounded is True
    assert g.expected_grounded is True


def test_validate_rag_ungrounded():
    q = _q("This sentence never appears in the lecture notes at all.")
    g = validate_question(q, source_text=SOURCE, use_rag=True)
    assert g.grounded is False
    assert g.match_type == "not_found"


def test_validate_baseline_empty_quote():
    q = _q("")
    g = validate_question(q, source_text=SOURCE, use_rag=False)
    assert g.grounded is False
    assert g.expected_grounded is False
    assert g.match_type == "baseline_empty"


def test_validate_quiz_metrics():
    questions = [
        _q("The round-robin algorithm uses a fixed time quantum."),
        _q("Hallucinated claim not in the document whatsoever."),
        _q(""),
    ]
    groundings, metrics = validate_quiz(
        questions, source_text=SOURCE, use_rag=True
    )
    assert metrics.total_questions == 3
    assert metrics.grounded_count == 1
    assert abs(metrics.quote_in_source_rate - 1 / 3) < 1e-9
    assert len(groundings) == 3


def test_filter_grounded_keeps_only_verified():
    questions = [
        _q("Process scheduling decides which ready process runs next."),
        _q("Completely invented material about quantum foam."),
    ]
    groundings, _ = validate_quiz(questions, source_text=SOURCE, use_rag=True)
    kept_q, kept_g = filter_grounded(questions, groundings)
    assert len(kept_q) == 1
    assert kept_g[0].grounded is True
