"""Unit tests for source-quote grounding validation (no API key required)."""

from app.generator import QuizQuestion
from app.grounding import (
    aggregate_soft_badges,
    annotate_exam_paper_dict,
    annotate_note_answer_dict,
    filter_grounded,
    normalize_for_match,
    options_are_unique,
    quote_in_source,
    resolve_grounded_serving,
    soft_badge_for_quote,
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


def test_resolve_grounded_serving_partial_filter():
    questions = [
        _q("The round-robin algorithm uses a fixed time quantum."),
        _q("Completely invented material about quantum foam."),
    ]
    groundings, _ = validate_quiz(questions, source_text=SOURCE, use_rag=True)
    served_q, served_g, filtered, best = resolve_grounded_serving(
        questions,
        groundings,
        use_rag=True,
        require_grounding=True,
    )
    assert len(served_q) == 1
    assert filtered == 1
    assert best is False
    assert served_g[0].grounded is True


def test_resolve_grounded_serving_soft_fallback_when_none_grounded():
    questions = [
        _q("Hallucinated one never in notes."),
        _q("Another invented claim about unicorns."),
    ]
    groundings, _ = validate_quiz(questions, source_text=SOURCE, use_rag=True)
    assert all(not g.grounded for g in groundings)
    served_q, served_g, filtered, best = resolve_grounded_serving(
        questions,
        groundings,
        use_rag=True,
        require_grounding=True,
    )
    # Soft fallback: still serve all questions, flag best_effort.
    assert len(served_q) == 2
    assert filtered == 0
    assert best is True
    assert len(served_g) == 2


def test_resolve_grounded_serving_report_only_keeps_all():
    questions = [
        _q("The round-robin algorithm uses a fixed time quantum."),
        _q("Completely invented material about quantum foam."),
    ]
    groundings, _ = validate_quiz(questions, source_text=SOURCE, use_rag=True)
    served_q, _, filtered, best = resolve_grounded_serving(
        questions,
        groundings,
        use_rag=True,
        require_grounding=False,
    )
    assert len(served_q) == 2
    assert filtered == 0
    assert best is False


def test_soft_badge_verified_and_not_found():
    ok = soft_badge_for_quote(
        "The round-robin algorithm uses a fixed time quantum.",
        SOURCE,
    )
    assert ok["status"] == "verified"
    assert ok["grounded"] is True
    bad = soft_badge_for_quote("Invented claim about unicorns and rainbows.", SOURCE)
    assert bad["status"] == "not_found"
    empty = soft_badge_for_quote("", SOURCE)
    assert empty["status"] == "partial"
    assert empty["label"] == "No quote"
    short = soft_badge_for_quote("short", SOURCE)
    assert short["status"] == "partial"


def test_annotate_exam_paper_never_drops_parts():
    paper = {
        "questions": [
            {
                "number": 1,
                "heading": "QUESTION ONE",
                "parts": [
                    {
                        "label": "a",
                        "prompt": "Discuss RR",
                        "marks": 5,
                        "source_quote": "The round-robin algorithm uses a fixed time quantum.",
                    },
                    {
                        "label": "b",
                        "prompt": "Invented",
                        "marks": 3,
                        "source_quote": "This quote is totally not in the lecture notes.",
                    },
                    {
                        "label": "c",
                        "prompt": "No cite",
                        "marks": 2,
                        "source_quote": "",
                    },
                ],
            }
        ]
    }
    out = annotate_exam_paper_dict(paper, SOURCE)
    parts = out["questions"][0]["parts"]
    assert len(parts) == 3  # never dropped
    assert parts[0]["grounding"]["status"] == "verified"
    assert parts[1]["grounding"]["status"] == "not_found"
    assert parts[2]["grounding"]["status"] == "partial"
    summary = out["grounding_summary"]
    assert summary["total_parts"] == 3
    assert summary["verified"] == 1
    assert summary["not_found"] == 1
    assert summary["partial"] == 1


def test_annotate_note_answer_aggregate_partial():
    ans = {
        "question": "What is RR?",
        "full_answer": "…",
        "source_quotes": [
            "The round-robin algorithm uses a fixed time quantum.",
            "Completely invented material about quantum foam.",
        ],
    }
    out = annotate_note_answer_dict(ans, SOURCE)
    assert len(out["quote_groundings"]) == 2
    assert out["quote_groundings"][0]["status"] == "verified"
    assert out["quote_groundings"][1]["status"] == "not_found"
    assert out["grounding"]["status"] == "partial"
    # Aggregate helper alone
    agg = aggregate_soft_badges(out["quote_groundings"])
    assert agg["status"] == "partial"
    assert agg["verified_count"] == 1
