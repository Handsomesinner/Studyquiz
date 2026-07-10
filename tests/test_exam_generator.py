"""Lightweight tests for exam paper helpers (no API key)."""

from app.exam_generator import (
    ExamPaper,
    ExamPart,
    ExamQuestion,
    ExamSubPart,
    _guess_code,
    _guess_title,
    assign_chunk_slices,
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


def test_assign_chunk_slices_covers_document():
    chunks = [f"chunk {i} " + ("word " * 50) for i in range(40)]
    slices = assign_chunk_slices(chunks, 4, per_question=6)
    assert len(slices) == 4
    assert all(1 <= len(s) <= 6 for s in slices)
    # Different regions (first chunk of Q1 before first of last Q)
    assert slices[0][0].startswith("chunk 0")
    assert "chunk 3" in slices[-1][0] or "chunk 3" in slices[-1][-1] or int(
        slices[-1][0].split()[1]
    ) >= 20
