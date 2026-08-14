"""SQLite persistence tests (no API key, no network)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.generator import QuizQuestion
from app.grounding import QuestionGrounding
from app import store


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("STUDYQUIZ_DB", str(db))
    # Ensure local SQLite path (no remote Turso during unit tests).
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("LIBSQL_URL", raising=False)
    monkeypatch.delenv("STUDYQUIZ_TURSO_URL", raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    store.reset_connection_state()
    store.init_db()
    yield db
    store.reset_connection_state()


def _sample_question() -> QuizQuestion:
    return QuizQuestion(
        question="What does round-robin use?",
        options=["Time quantum", "Priority only", "FIFO only", "Stacks"],
        correct_index=0,
        explanation="From the notes.",
        source_quote="The round-robin algorithm uses a fixed time quantum.",
    )


def _sample_grounding() -> QuestionGrounding:
    return QuestionGrounding(
        grounded=True,
        match_type="exact",
        source_quote="The round-robin algorithm uses a fixed time quantum.",
        options_unique=True,
        expected_grounded=True,
    )


def test_save_and_get_document(tmp_db):
    store.save_document(
        doc_id="doc1",
        title="os.pdf",
        text="Process scheduling decides which ready process runs next.",
        chunks=["Process scheduling decides which ready process runs next."],
    )
    doc = store.get_document("doc1")
    assert doc is not None
    assert doc["title"] == "os.pdf"
    assert len(doc["chunks"]) == 1
    assert doc["retriever"] is not None
    hits = doc["retriever"].search("scheduling", top_k=1)
    assert hits  # BM25 rebuilt from chunks

    listed = store.list_documents()
    assert len(listed) == 1
    assert listed[0]["id"] == "doc1"
    assert listed[0]["num_chunks"] == 1


def test_document_survives_reconnect(tmp_db):
    store.save_document(
        doc_id="doc2",
        title="notes.txt",
        text="hello world " * 50,
        chunks=["hello world " * 40, "second chunk about memory"],
    )
    # Simulate restart: drop connection
    store.reset_connection_state()

    doc = store.get_document("doc2")
    assert doc is not None
    assert doc["title"] == "notes.txt"
    assert len(doc["chunks"]) == 2


def test_save_quiz_and_attempt(tmp_db):
    store.save_document(
        doc_id="doc3",
        title="t.pdf",
        text="abc",
        chunks=["abc"],
    )
    q = _sample_question()
    g = _sample_grounding()
    store.save_quiz(
        quiz_id="quiz1",
        document_id="doc3",
        use_rag=True,
        topic="scheduling",
        questions=[q],
        groundings=[g],
        pre_filter_metrics={"quote_in_source_rate": 1.0},
        served_metrics={"quote_in_source_rate": 1.0},
        context_chunks=["chunk text"],
    )
    quiz = store.get_quiz("quiz1")
    assert quiz is not None
    assert quiz["use_rag"] is True
    assert quiz["topic"] == "scheduling"
    assert quiz["questions"][0].question == q.question
    assert quiz["groundings"][0].grounded is True

    store.save_attempt(
        attempt_id="att1",
        quiz_id="quiz1",
        answers=[0],
        score=1,
        total=1,
        results=[{"correct": True}],
    )
    # No public getter required; just ensure no error and FK works.


def test_eval_rows_persist(tmp_db):
    store.append_eval_rows(
        [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "quiz_id": "q1",
                "document_id": "d1",
                "document_title": "x.pdf",
                "use_rag": True,
                "topic": "",
                "phase": "pre_filter",
                "question_index": 0,
                "question": "Q?",
                "options": "A | B | C | D",
                "correct_index": 0,
                "source_quote": "quote",
                "grounded": True,
                "match_type": "exact",
                "options_unique": True,
                "expected_grounded": True,
            }
        ]
    )
    store.reset_connection_state()

    rows = store.list_eval_rows(phase="pre_filter")
    assert len(rows) == 1
    assert rows[0]["grounded"] is True
    assert rows[0]["use_rag"] is True


def test_storage_info_local_sqlite(tmp_db):
    info = store.storage_info()
    assert info["backend"] == "sqlite"
    assert info["durable"] is True
    assert "test.db" in info["db_path"] or str(tmp_db) in info["db_path"]


def test_storage_info_turso(monkeypatch):
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://studyquiz-demo.turso.io")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "fake-token")
    store.reset_connection_state()
    assert store.using_turso() is True
    info = store.storage_info()
    assert info["backend"] == "turso"
    assert info["durable"] is True
    assert "studyquiz-demo.turso.io" in info["host"]
    store.reset_connection_state()


def test_turso_arg_encoding():
    assert store._encode_turso_arg(None) == {"type": "null"}
    assert store._encode_turso_arg(3) == {"type": "integer", "value": "3"}
    assert store._encode_turso_arg("hi") == {"type": "text", "value": "hi"}
    assert store._decode_turso_value({"type": "text", "value": "hi"}) == "hi"
    assert store._decode_turso_value({"type": "integer", "value": "7"}) == 7


def test_exam_paper_roundtrip(tmp_db):
    store.save_document(
        doc_id="doc-exam",
        title="sec.pdf",
        text="security notes",
        chunks=["security notes"],
    )
    store.save_exam_paper(
        exam_id="exam1",
        document_id="doc-exam",
        use_rag=True,
        topic=None,
        difficulty="medium",
        paper={"questions": [{"number": 1, "heading": "QUESTION ONE"}]},
        context_chunks=["security notes"],
    )
    row = store.get_exam_paper("exam1")
    assert row is not None
    assert row["paper"]["questions"][0]["heading"] == "QUESTION ONE"
    store.update_exam_paper(
        "exam1",
        {"questions": [{"number": 1, "heading": "QUESTION ONE", "guides": True}]},
    )
    row2 = store.get_exam_paper("exam1")
    assert row2["paper"]["questions"][0]["guides"] is True
