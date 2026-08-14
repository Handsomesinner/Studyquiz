"""Topic map + near-duplicate quality helpers."""

from app.generator import QuizQuestion
from app.grounding import QuestionGrounding
from app.quality import filter_near_duplicates, is_near_duplicate, jaccard, ungrounded_indices
from app.topics import extract_heading_topics, extract_phrase_topics, topic_map


def test_heading_topics():
    text = """
# Introduction to Networking
## IPSec and VPNs
### Key exchange protocols

Some body text about networking that is not a heading.
"""
    heads = extract_heading_topics(text)
    joined = " ".join(heads).lower()
    assert "networking" in joined or "ipsec" in joined


def test_phrase_topics_finds_repeated_bigrams():
    # Repeat distinctive phrases many times
    block = "virtual memory " * 20 + "page replacement " * 15 + "the the the " * 10
    phrases = extract_phrase_topics(block, limit=8)
    assert any("virtual" in p.lower() for p in phrases)


def test_topic_map_combines():
    text = "# Cryptography Basics\n\n" + ("public key " * 30) + ("private key " * 25)
    m = topic_map(text, max_topics=10)
    assert m
    assert all("topic" in x and "source" in x for x in m)


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
        _q("What is deadlock?"),  # exact-ish dup of first
    ]
    # force first and third to be near-dup
    qs[2] = _q("What is deadlock?")
    kept, _, dropped = filter_near_duplicates(qs)
    assert dropped >= 1
    assert len(kept) < len(qs)
    assert kept[0].question == "What is deadlock?"


def test_ungrounded_indices():
    gs = [
        QuestionGrounding(True, "exact", "q", True, True),
        QuestionGrounding(False, "not_found", "q", True, True),
        QuestionGrounding(False, "empty_quote", "", True, True),
    ]
    assert ungrounded_indices(gs, use_rag=True) == [1, 2]
    assert ungrounded_indices(gs, use_rag=False) == []
