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
    # Reset thread-local connection + schema flag so a new path is used.
    conn = getattr(store._local, "conn", None)
    if conn is not None:
        conn.close()
    store._local.conn = None
    store._schema_ready = False
    store.init_db()
    yield db
    conn = getattr(store._local, "conn", None)
    if conn is not None:
        conn.close()
        store._local.conn = None
    store._schema_ready = False


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
    store._local.conn.close()
    store._local.conn = None
    store._schema_ready = False

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
    store._local.conn.close()
    store._local.conn = None
    store._schema_ready = False

    rows = store.list_eval_rows(phase="pre_filter")
    assert len(rows) == 1
    assert rows[0]["grounded"] is True
    assert rows[0]["use_rag"] is True
