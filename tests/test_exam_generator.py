"""Lightweight tests for exam paper helpers (no API key)."""

from app.exam_generator import (
    ExamPaper,
    ExamPart,
    ExamQuestion,
    ExamSubPart,
    _guess_code,
    _guess_title,
    assign_chunk_slices,
    build_exam_generation_meta,
    paper_total_marks,
    renumber_exam_questions,
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


def test_build_exam_generation_meta_complete():
    meta = build_exam_generation_meta(
        requested=4,
        generated=4,
        failed_slots=[],
        retried_slots=[],
        errors=[],
    )
    assert meta["partial"] is False
    assert meta["message"] is None
    assert meta["missing_count"] == 0


def test_build_exam_generation_meta_partial_message():
    meta = build_exam_generation_meta(
        requested=4,
        generated=3,
        failed_slots=[2],
        retried_slots=[2],
        errors=["QUESTION 2: timeout"],
    )
    assert meta["partial"] is True
    assert meta["generated"] == 3
    assert meta["requested"] == 4
    assert meta["missing_count"] == 1
    assert "3 of 4" in meta["message"]
    assert "QUESTION TWO" in meta["message"]
    assert "retry" in meta["message"].lower()


def test_renumber_exam_questions_consecutive():
    qs = [
        ExamQuestion(
            number=1,
            heading="QUESTION ONE",
            parts=[ExamPart(label="a", prompt="x", marks=5)],
        ),
        ExamQuestion(
            number=3,
            heading="QUESTION THREE",
            parts=[ExamPart(label="a", prompt="y", marks=5)],
        ),
    ]
    renumber_exam_questions(qs)
    assert qs[0].number == 1 and qs[0].heading == "QUESTION ONE"
    assert qs[1].number == 2 and qs[1].heading == "QUESTION TWO"
