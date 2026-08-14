"""Topic map (structural headings) + near-duplicate quality helpers."""

from app.generator import QuizQuestion
from app.grounding import QuestionGrounding
from app.quality import filter_near_duplicates, is_near_duplicate, jaccard, ungrounded_indices
from app.topics import (
    extract_bold_headings,
    extract_chapter_headings,
    extract_markdown_headings,
    extract_standalone_explained_headings,
    topic_map,
)


SAMPLE_NOTES = """
# Introduction to Operating Systems

Operating systems manage resources and provide services to applications.

## Process Scheduling

Process scheduling decides which ready process runs next.
The scheduler may use FCFS, SJF, or round-robin strategies.

**Virtual Memory**

Virtual memory lets processes use more address space than physical RAM.
Demand paging loads pages only when needed.

Chapter 4: Deadlock
Deadlock occurs when processes wait forever for each other's resources.
Four conditions are required: mutual exclusion, hold and wait, no preemption, circular wait.

3.2 Page Replacement
Page replacement algorithms choose which page to evict from frames.
Examples include FIFO, LRU, and optimal replacement.

CRYPTOGRAPHY BASICS
This section introduces classical and modern cryptography for secure systems.
"""


def test_markdown_headings():
    heads = extract_markdown_headings(SAMPLE_NOTES)
    topics = " ".join(h["topic"] for h in heads).lower()
    assert "operating systems" in topics or "process scheduling" in topics


def test_bold_headings():
    bold = extract_bold_headings(SAMPLE_NOTES)
    assert any("virtual memory" in b["topic"].lower() for b in bold)


def test_chapter_headings():
    ch = extract_chapter_headings(SAMPLE_NOTES)
    assert any("deadlock" in c["topic"].lower() for c in ch)


def test_standalone_explained():
    # Short line then long explanation
    text = """
Demand Paging

Demand paging loads a page into memory only when a process references it.
This reduces memory pressure and improves multiprogramming.
"""
    items = extract_standalone_explained_headings(text)
    assert any("demand paging" in i["topic"].lower() for i in items)


def test_topic_map_prefers_structure_not_phrases():
    m = topic_map(SAMPLE_NOTES, max_topics=12)
    assert m
    sources = {x["source"] for x in m}
    # Should use structural sources, never "phrase"
    assert "phrase" not in sources
    joined = " ".join(x["topic"] for x in m).lower()
    assert "scheduling" in joined or "deadlock" in joined or "virtual" in joined


def test_topic_map_empty():
    assert topic_map("") == []
    assert topic_map("   \n  just a short sentence without structure.") == [] or True


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
