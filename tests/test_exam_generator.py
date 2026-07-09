"""Lightweight tests for exam paper helpers (no API key)."""

from app.exam_generator import (
    ExamPaper,
    ExamPart,
    ExamQuestion,
    ExamSubPart,
    _guess_code,
    _guess_title,
    paper_total_marks,
)


def test_guess_code_from_filename():
    assert "CSC" in _guess_code("CSC409_lecture_notes.pdf").upper() or _guess_code(
        "notes.pdf"
    ) == "CSC 000"


def test_guess_title_strips_extension():
    assert "security" in _guess_title("Computer_Information_Security.pdf").lower()


def test_paper_total_marks():
    paper = ExamPaper(
        instructions=["Answer Q1 and any other."],
        questions=[
            ExamQuestion(
                number=1,
                heading="QUESTION ONE",
                parts=[
                    ExamPart(label="a", prompt="Discuss DOS", marks=6, subparts=[]),
                    ExamPart(
                        label="b",
                        prompt="Define terms",
                        marks=4,
                        subparts=[
                            ExamSubPart(label="i", text="Threat"),
                            ExamSubPart(label="ii", text="Vulnerability"),
                        ],
                    ),
                ],
            ),
            ExamQuestion(
                number=2,
                heading="QUESTION TWO",
                parts=[ExamPart(label="a", prompt="Six Ps", marks=3)],
            ),
        ],
    )
    assert paper_total_marks(paper) == 13
