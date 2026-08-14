"""Near-duplicate quality helpers (no topic map)."""

from app.generator import QuizQuestion
from app.grounding import QuestionGrounding
from app.quality import filter_near_duplicates, is_near_duplicate, jaccard, ungrounded_indices


def _q(stem: str, opts=None) -> QuizQuestion:
    return QuizQuestion(
        question=stem,
        options=opts or ["A", "B", "C", "D"],
        correct_index=0,
        explanation="e",
        source_quote="quote here is long enough",
    )


def test_jaccard_and_near_dup():
    assert jaccard("process scheduling round robin", "process scheduling fifo") > 0.3
    a = _q("What is the purpose of a page table in virtual memory?")
    b = _q("What is the purpose of a page table in virtual memory systems?")
    assert is_near_duplicate(a, b)


def test_filter_near_duplicates_keeps_first():
    qs = [
        _q("What is deadlock?"),
        _q("What is deadlock prevention?"),
        _q("What is deadlock?"),
    ]
    kept, _, dropped = filter_near_duplicates(qs)
    assert dropped >= 1
    assert len(kept) < len(qs)


def test_ungrounded_indices():
    gs = [
        QuestionGrounding(True, "exact", "q", True, True),
        QuestionGrounding(False, "not_found", "q", True, True),
        QuestionGrounding(False, "empty_quote", "", True, True),
    ]
    assert ungrounded_indices(gs, use_rag=True) == [1, 2]
    assert ungrounded_indices(gs, use_rag=False) == []
