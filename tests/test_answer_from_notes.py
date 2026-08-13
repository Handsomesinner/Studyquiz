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
