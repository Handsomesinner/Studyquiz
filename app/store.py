"""SQLite persistence for documents, quizzes, attempts, and evaluation rows.

Replaces the in-memory dicts so demos survive process restarts. Uses only the
stdlib ``sqlite3`` module — no extra dependencies. BM25 indexes are rebuilt
from stored chunks on load (cheap for lecture-sized documents).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .generator import QuizQuestion
from .grounding import QuestionGrounding
from .retriever import BM25Retriever

# Default (local): <repo>/data/studyquiz.db  — override with STUDYQUIZ_DB.
# On Vercel/Lambda the deployment FS is read-only; only /tmp is writable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = _REPO_ROOT / "data" / "studyquiz.db"

_local = threading.local()
_schema_ready = False
_schema_lock = threading.Lock()


def _running_serverless() -> bool:
    return bool(
        os.getenv("VERCEL")
        or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
        or os.getenv("VERCEL_ENV")
    )


def db_path() -> Path:
    raw = os.getenv("STUDYQUIZ_DB")
    if raw:
        return Path(raw)
    if _running_serverless():
        return Path("/tmp/studyquiz.db")
    return DEFAULT_DB_PATH


def _connect() -> sqlite3.Connection:
    path = db_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only parent — fall back to /tmp so the function still starts.
        path = Path("/tmp/studyquiz.db")
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent CREATE TABLE — safe to call on every cold start."""
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        _schema_ready = True


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """Per-thread connection (safe with uvicorn workers / reload)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    _ensure_schema(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    chunks_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quizzes (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    use_rag INTEGER NOT NULL,
    topic TEXT,
    questions_json TEXT NOT NULL,
    groundings_json TEXT NOT NULL,
    pre_filter_metrics_json TEXT NOT NULL,
    served_metrics_json TEXT NOT NULL,
    context_chunks_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id TEXT PRIMARY KEY,
    quiz_id TEXT NOT NULL,
    answers_json TEXT NOT NULL,
    score INTEGER NOT NULL,
    total INTEGER NOT NULL,
    results_json TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    FOREIGN KEY (quiz_id) REFERENCES quizzes(id)
);

CREATE TABLE IF NOT EXISTS eval_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    quiz_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_title TEXT,
    use_rag INTEGER NOT NULL,
    topic TEXT,
    phase TEXT NOT NULL,
    question_index INTEGER NOT NULL,
    question TEXT NOT NULL,
    options TEXT NOT NULL,
    correct_index INTEGER NOT NULL,
    source_quote TEXT,
    grounded INTEGER NOT NULL,
    match_type TEXT NOT NULL,
    options_unique INTEGER NOT NULL,
    expected_grounded INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quizzes_doc ON quizzes(document_id);
CREATE INDEX IF NOT EXISTS idx_eval_quiz ON eval_rows(quiz_id);
CREATE INDEX IF NOT EXISTS idx_eval_phase ON eval_rows(phase);
"""


def init_db() -> None:
    """Create tables if they do not exist (also runs lazily on first query)."""
    with connection() as _conn:
        pass  # connection() already calls _ensure_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _loads(raw: str) -> Any:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def save_document(
    *,
    doc_id: str,
    title: str,
    text: str,
    chunks: list[str],
) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (id, title, text, chunks_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, title or "untitled", text, _dumps(chunks), _now()),
        )


def get_document(doc_id: str) -> dict | None:
    """Return {id, title, text, chunks, retriever} or None."""
    with connection() as conn:
        row = conn.execute(
            "SELECT id, title, text, chunks_json, created_at FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    if row is None:
        return None
    chunks = _loads(row["chunks_json"])
    retriever = BM25Retriever()
    retriever.index(chunks)
    return {
        "id": row["id"],
        "title": row["title"],
        "text": row["text"],
        "chunks": chunks,
        "retriever": retriever,
        "created_at": row["created_at"],
    }


def list_documents() -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, chunks_json, created_at
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()
    out = []
    for row in rows:
        chunks = _loads(row["chunks_json"])
        out.append(
            {
                "id": row["id"],
                "title": row["title"],
                "num_chunks": len(chunks),
                "created_at": row["created_at"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Quizzes
# ---------------------------------------------------------------------------


def _question_to_dict(q: QuizQuestion) -> dict:
    return q.model_dump()


def _question_from_dict(d: dict) -> QuizQuestion:
    return QuizQuestion.model_validate(d)


def _grounding_to_dict(g: QuestionGrounding) -> dict:
    return g.to_dict()


def _grounding_from_dict(d: dict) -> QuestionGrounding:
    return QuestionGrounding(
        grounded=bool(d["grounded"]),
        match_type=d["match_type"],
        source_quote=d.get("source_quote") or "",
        options_unique=bool(d["options_unique"]),
        expected_grounded=bool(d["expected_grounded"]),
    )


def save_quiz(
    *,
    quiz_id: str,
    document_id: str,
    use_rag: bool,
    topic: str | None,
    questions: list[QuizQuestion],
    groundings: list[QuestionGrounding],
    pre_filter_metrics: dict,
    served_metrics: dict,
    context_chunks: list[str],
) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO quizzes (
                id, document_id, use_rag, topic,
                questions_json, groundings_json,
                pre_filter_metrics_json, served_metrics_json,
                context_chunks_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                quiz_id,
                document_id,
                int(use_rag),
                topic,
                _dumps([_question_to_dict(q) for q in questions]),
                _dumps([_grounding_to_dict(g) for g in groundings]),
                _dumps(pre_filter_metrics),
                _dumps(served_metrics),
                _dumps(context_chunks),
                _now(),
            ),
        )


def get_quiz(quiz_id: str) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM quizzes WHERE id = ?",
            (quiz_id,),
        ).fetchone()
    if row is None:
        return None
    questions = [_question_from_dict(d) for d in _loads(row["questions_json"])]
    groundings = [_grounding_from_dict(d) for d in _loads(row["groundings_json"])]
    return {
        "id": row["id"],
        "questions": questions,
        "groundings": groundings,
        "doc_id": row["document_id"],
        "use_rag": bool(row["use_rag"]),
        "topic": row["topic"],
        "pre_filter_metrics": _loads(row["pre_filter_metrics_json"]),
        "served_metrics": _loads(row["served_metrics_json"]),
        "context_chunks": _loads(row["context_chunks_json"]),
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# Attempts (scores)
# ---------------------------------------------------------------------------


def save_attempt(
    *,
    attempt_id: str,
    quiz_id: str,
    answers: list[int],
    score: int,
    total: int,
    results: list[dict],
) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO quiz_attempts (
                id, quiz_id, answers_json, score, total, results_json, submitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                quiz_id,
                _dumps(answers),
                score,
                total,
                _dumps(results),
                _now(),
            ),
        )


# ---------------------------------------------------------------------------
# Evaluation rows
# ---------------------------------------------------------------------------


def append_eval_rows(rows: list[dict]) -> None:
    if not rows:
        return
    with connection() as conn:
        conn.executemany(
            """
            INSERT INTO eval_rows (
                timestamp, quiz_id, document_id, document_title,
                use_rag, topic, phase, question_index, question, options,
                correct_index, source_quote, grounded, match_type,
                options_unique, expected_grounded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["timestamp"],
                    r["quiz_id"],
                    r["document_id"],
                    r.get("document_title") or "",
                    int(bool(r["use_rag"])),
                    r.get("topic") or "",
                    r["phase"],
                    int(r["question_index"]),
                    r["question"],
                    r["options"],
                    int(r["correct_index"]),
                    r.get("source_quote") or "",
                    int(bool(r["grounded"])),
                    r["match_type"],
                    int(bool(r["options_unique"])),
                    int(bool(r["expected_grounded"])),
                )
                for r in rows
            ],
        )


def list_eval_rows(*, phase: str | None = None) -> list[dict]:
    with connection() as conn:
        if phase:
            rows = conn.execute(
                "SELECT * FROM eval_rows WHERE phase = ? ORDER BY id",
                (phase,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM eval_rows ORDER BY id"
            ).fetchall()
    return [_eval_row_to_dict(r) for r in rows]


def _eval_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "timestamp": row["timestamp"],
        "quiz_id": row["quiz_id"],
        "document_id": row["document_id"],
        "document_title": row["document_title"],
        "use_rag": bool(row["use_rag"]),
        "topic": row["topic"],
        "phase": row["phase"],
        "question_index": row["question_index"],
        "question": row["question"],
        "options": row["options"],
        "correct_index": row["correct_index"],
        "source_quote": row["source_quote"],
        "grounded": bool(row["grounded"]),
        "match_type": row["match_type"],
        "options_unique": bool(row["options_unique"]),
        "expected_grounded": bool(row["expected_grounded"]),
    }
