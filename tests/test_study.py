"""Study loop helpers and store (SRS, banks, wrong items)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import store
from app.generator import QuizQuestion
from app.grounding import QuestionGrounding
from app.study import fingerprint_question, sm2_schedule


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "study.db"
    monkeypatch.setenv("STUDYQUIZ_DB", str(db))
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("LIBSQL_URL", raising=False)
    store.reset_connection_state()
    store.init_db()
    yield db
    store.reset_connection_state()


def test_fingerprint_stable():
    a = fingerprint_question("  What is  RAM? ")
    b = fingerprint_question("what is ram?")
    assert a == b
    assert a != fingerprint_question("What is CPU?")


def test_sm2_again_due_now():
    s = sm2_schedule(grade="again", ease=2.5, interval_days=5, reps=3)
    assert s["reps"] == 0
    assert s["interval_days"] == 0
    assert s["lapses"] == 1


def test_sm2_good_increases_interval():
    s1 = sm2_schedule(grade="good", ease=2.5, interval_days=0, reps=0)
    assert s1["interval_days"] >= 1
    s2 = sm2_schedule(
        grade="good",
        ease=s1["ease"],
        interval_days=s1["interval_days"],
        reps=s1["reps"],
    )
    assert s2["interval_days"] >= s1["interval_days"]


def test_srs_upsert_and_due(tmp_db):
    store.save_document(
        doc_id="d1", title="t.pdf", text="hello", chunks=["hello"]
    )
    store.upsert_srs_card(
        card_id="c1",
        document_id="d1",
        fingerprint="fp1",
        card_type="mcq",
        payload={"question": "Q?", "options": ["a", "b", "c", "d"], "correct_index": 0},
        due_at="2000-01-01T00:00:00+00:00",
    )
    due = store.list_due_srs_cards("d1", limit=5)
    assert len(due) == 1
    assert due[0]["payload"]["question"] == "Q?"
    store.upsert_srs_card(
        card_id="c1",
        document_id="d1",
        fingerprint="fp1",
        card_type="mcq",
        payload={"question": "Q2?", "options": ["a", "b", "c", "d"], "correct_index": 1},
        due_at="2000-01-01T00:00:00+00:00",
    )
    due2 = store.list_due_srs_cards("d1")
    assert len(due2) == 1
    assert due2[0]["payload"]["question"] == "Q2?"


def test_question_bank(tmp_db):
    store.save_question_bank(
        bank_id="b1",
        title="Midterm",
        questions=["What is a process?", "Define deadlock."],
        document_id=None,
    )
    listed = store.list_question_banks()
    assert listed[0]["question_count"] == 2
    got = store.get_question_bank("b1")
    assert len(got["questions"]) == 2
    assert store.rename_question_bank("b1", "Final") is True
    assert store.get_question_bank("b1")["title"] == "Final"
    assert store.delete_question_bank("b1") is True
    assert store.get_question_bank("b1") is None


def test_list_wrong_items(tmp_db):
    store.save_document(doc_id="d2", title="n.pdf", text="x", chunks=["x"])
    q = QuizQuestion(
        question="What is X?",
        options=["A", "B", "C", "D"],
        correct_index=0,
        explanation="Because.",
        source_quote="x",
    )
    g = QuestionGrounding(
        grounded=True,
        match_type="exact",
        source_quote="x",
        options_unique=True,
        expected_grounded=True,
    )
    store.save_quiz(
        quiz_id="qz1",
        document_id="d2",
        use_rag=True,
        topic=None,
        questions=[q],
        groundings=[g],
        pre_filter_metrics={},
        served_metrics={},
        context_chunks=[],
    )
    store.save_attempt(
        attempt_id="a1",
        quiz_id="qz1",
        answers=[1],
        score=0,
        total=1,
        results=[
            {
                "correct": False,
                "chosen_index": 1,
                "correct_index": 0,
                "explanation": "Because.",
                "source_quote": "x",
                "grounded": True,
                "match_type": "exact",
            }
        ],
    )
    wrong = store.list_wrong_items("d2")
    assert len(wrong) == 1
    assert wrong[0]["payload"]["question"] == "What is X?"


def test_exam_attempt_roundtrip(tmp_db):
    store.save_document(doc_id="d3", title="e.pdf", text="y", chunks=["y"])
    store.save_exam_paper(
        exam_id="ex1",
        document_id="d3",
        use_rag=True,
        topic=None,
        difficulty="medium",
        paper={"questions": []},
        context_chunks=[],
    )
    store.save_exam_attempt(
        attempt_id="ea1",
        exam_id="ex1",
        answers={"1-a": "My answer"},
        status="draft",
    )
    d = store.get_exam_attempt("ea1")
    assert d["answers"]["1-a"] == "My answer"
    store.save_exam_attempt(
        attempt_id="ea1",
        exam_id="ex1",
        answers={"1-a": "Updated"},
        status="submitted",
    )
    d2 = store.get_latest_exam_draft("ex1")
    assert d2["status"] == "submitted"
    assert d2["answers"]["1-a"] == "Updated"


def test_mixed_and_share(tmp_db):
    store.save_document(doc_id="d4", title="m.pdf", text="z", chunks=["z"])
    store.save_mixed_paper(
        mixed_id="mx1",
        document_id="d4",
        title="Mix",
        config={"sections": [{"type": "mcq", "quiz_id": "q"}]},
    )
    m = store.get_mixed_paper("mx1")
    assert m["title"] == "Mix"
    store.save_share_token(
        token="tok1",
        resource_type="mixed",
        resource_id="mx1",
        password_hash=None,
    )
    s = store.get_share_token("tok1")
    assert s["resource_id"] == "mx1"
