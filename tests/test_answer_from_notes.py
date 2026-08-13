"""Tests for parsing user-supplied questions (no API key)."""

from app.answer_from_notes import parse_questions


def test_parse_numbered_list():
    text = """
1. What is a queue?
2. Define throughput.
3. Explain Little's law.
"""
    qs = parse_questions(text)
    assert len(qs) == 3
    assert "queue" in qs[0].lower()
    assert "throughput" in qs[1].lower()


def test_parse_blank_line_blocks():
    text = """Discuss deadlock prevention.

Compare paging and segmentation.
"""
    qs = parse_questions(text)
    assert len(qs) == 2


def test_parse_question_prefix():
    text = """
Question 1: What is virtual memory?
Question 2: List two page replacement algorithms.
"""
    qs = parse_questions(text)
    assert len(qs) >= 2


def test_parse_dedupes():
    text = """1. What is CPU scheduling?
1. What is CPU scheduling?
"""
    qs = parse_questions(text)
    assert len(qs) == 1


def test_parse_empty():
    assert parse_questions("") == []
    assert parse_questions("   \n  ") == []


def test_parse_lettered_parts():
    text = """
a) Discuss DOS attacks.
b) Explain DNS poisoning.
c) What is cross-site scripting?
d) Define industrial espionage.
e) How does SSL protect users?
"""
    qs = parse_questions(text)
    assert len(qs) >= 5


def test_parse_many_numbered():
    lines = [f"{i}. Question number {i} about operating systems?" for i in range(1, 11)]
    qs = parse_questions("\n".join(lines))
    assert len(qs) == 10
