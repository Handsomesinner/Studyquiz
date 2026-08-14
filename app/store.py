"""Persistence for documents, quizzes, attempts, evaluation rows, and exams.

**Local demos** — stdlib ``sqlite3`` file DB (``data/studyquiz.db`` by default).

**Vercel / durable** — when ``TURSO_DATABASE_URL`` (and usually
``TURSO_AUTH_TOKEN``) are set, the same schema is stored on **Turso**
(libSQL over HTTPS via ``httpx``). Data then survives cold starts and is
shared across serverless instances.

Without Turso on Vercel, falls back to ``/tmp/studyquiz.db`` (ephemeral).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import httpx

from .generator import QuizQuestion
from .grounding import QuestionGrounding
from .retriever import BM25Retriever

# Default (local): <repo>/data/studyquiz.db  — override with STUDYQUIZ_DB.
# On Vercel/Lambda without Turso the deployment FS is read-only; only /tmp.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = _REPO_ROOT / "data" / "studyquiz.db"

_local = threading.local()
_schema_ready = False
_schema_lock = threading.Lock()

# Turso HTTP pipeline client (shared; thread-safe enough for short requests).
_turso_client: httpx.Client | None = None
_turso_client_lock = threading.Lock()


def _running_serverless() -> bool:
    return bool(
        os.getenv("VERCEL")
        or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
        or os.getenv("VERCEL_ENV")
    )


def turso_url() -> str | None:
    """Remote libSQL / Turso URL if configured."""
    raw = (
        os.getenv("TURSO_DATABASE_URL")
        or os.getenv("LIBSQL_URL")
        or os.getenv("STUDYQUIZ_TURSO_URL")
        or ""
    ).strip()
    return raw or None


def turso_auth_token() -> str:
    return (
        os.getenv("TURSO_AUTH_TOKEN")
        or os.getenv("LIBSQL_AUTH_TOKEN")
        or os.getenv("STUDYQUIZ_TURSO_TOKEN")
        or ""
    ).strip()


def using_turso() -> bool:
    return turso_url() is not None


def storage_backend() -> str:
    """``turso`` | ``sqlite``."""
    return "turso" if using_turso() else "sqlite"


def storage_is_durable() -> bool:
    """True when data is expected to survive process/instance restarts."""
    if using_turso():
        return True
    # Local file DB is durable for that machine; /tmp on serverless is not.
    if _running_serverless() and not using_turso():
        return False
    return True


def db_path() -> Path:
    """Local SQLite path (ignored when Turso is active)."""
    raw = os.getenv("STUDYQUIZ_DB")
    if raw:
        return Path(raw)
    if _running_serverless():
        return Path("/tmp/studyquiz.db")
    return DEFAULT_DB_PATH


def storage_info() -> dict[str, Any]:
    """Describe active persistence for health checks / UI."""
    if using_turso():
        url = turso_url() or ""
        # Hide credentials; show host only.
        host = url
        for prefix in ("libsql://", "https://", "http://"):
            if host.startswith(prefix):
                host = host[len(prefix) :]
                break
        host = host.split("/")[0]
        return {
            "backend": "turso",
            "durable": True,
            "host": host,
            "db_path": f"turso://{host}",
            "hint": "Documents and quizzes persist across deploys and cold starts.",
        }
    path = str(db_path())
    durable = storage_is_durable()
    return {
        "backend": "sqlite",
        "durable": durable,
        "host": None,
        "db_path": path,
        "hint": (
            "Local SQLite file — data survives restarts on this machine."
            if durable
            else "Ephemeral /tmp SQLite on Vercel — set TURSO_DATABASE_URL "
            "and TURSO_AUTH_TOKEN for durable storage."
        ),
    }


def _turso_pipeline_url() -> str:
    url = turso_url() or ""
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://") :]
    elif url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    url = url.rstrip("/")
    if not url.endswith("/v2/pipeline"):
        url = url + "/v2/pipeline"
    return url


def _http_client() -> httpx.Client:
    global _turso_client
    if _turso_client is not None:
        return _turso_client
    with _turso_client_lock:
        if _turso_client is None:
            _turso_client = httpx.Client(timeout=60.0)
        return _turso_client


def _encode_turso_arg(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        # SQLite has no bool; store 0/1 like the local path.
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": str(value)}
    if isinstance(value, (bytes, bytearray)):
        import base64

        return {
            "type": "blob",
            "base64": base64.b64encode(bytes(value)).decode("ascii"),
        }
    return {"type": "text", "value": str(value)}


def _decode_turso_value(cell: Any) -> Any:
    """Normalize a cell from the Turso v2 pipeline response."""
    if cell is None:
        return None
    if isinstance(cell, dict):
        # {"type":"text","value":"..."} or {"type":"null"}
        t = cell.get("type")
        if t == "null" or "value" not in cell and t != "blob":
            return None
        if t == "integer":
            try:
                return int(cell["value"])
            except (TypeError, ValueError):
                return cell.get("value")
        if t == "float":
            try:
                return float(cell["value"])
            except (TypeError, ValueError):
                return cell.get("value")
        if t == "blob":
            return cell.get("base64")
        return cell.get("value")
    return cell


class _DictRow(dict):
    """dict that also allows sqlite3.Row-style ``row['col']`` access."""

    pass


class _TursoCursor:
    def __init__(self, columns: list[str], rows: list[list[Any]], rowcount: int = -1):
        self._columns = columns
        self._rows = rows
        self._i = 0
        self.rowcount = rowcount

    def fetchone(self) -> _DictRow | None:
        if self._i >= len(self._rows):
            return None
        vals = self._rows[self._i]
        self._i += 1
        return _DictRow(zip(self._columns, vals))

    def fetchall(self) -> list[_DictRow]:
        out = []
        while True:
            row = self.fetchone()
            if row is None:
                break
            out.append(row)
        return out


class TursoConnection:
    """Minimal sqlite3-like connection over Turso's HTTP pipeline API."""

    def __init__(self) -> None:
        self._pipeline = _turso_pipeline_url()
        self._token = turso_auth_token()
        self._pending: list[dict[str, Any]] = []

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _run_pipeline(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        body = {"requests": requests + [{"type": "close"}]}
        try:
            resp = _http_client().post(
                self._pipeline, headers=self._headers(), json=body
            )
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Turso request failed: {type(e).__name__}: {e}. "
                "Check TURSO_DATABASE_URL and network access."
            ) from e
        if resp.status_code >= 400:
            snippet = (resp.text or "")[:400]
            raise RuntimeError(
                f"Turso HTTP {resp.status_code}: {snippet or 'no body'}. "
                "Check TURSO_DATABASE_URL and TURSO_AUTH_TOKEN."
            )
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError("Turso returned non-JSON response.") from e
        results = data.get("results") or []
        # Each result: {"type":"ok","response":{...}} or {"type":"error",...}
        for item in results:
            if item.get("type") == "error":
                err = item.get("error") or item
                raise RuntimeError(f"Turso SQL error: {err}")
            # Nested error in ok response
            resp_body = item.get("response") or {}
            if isinstance(resp_body, dict) and resp_body.get("type") == "error":
                raise RuntimeError(f"Turso SQL error: {resp_body.get('error')}")
        return results

    def _stmt(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any]:
        stmt: dict[str, Any] = {"sql": sql}
        if params:
            stmt["args"] = [_encode_turso_arg(p) for p in params]
        return {"type": "execute", "stmt": stmt}

    def _parse_execute_result(self, item: dict[str, Any]) -> _TursoCursor:
        # Shape (HTTP v2):
        # { "type":"ok", "response": { "type":"execute", "result": { cols, rows, ... } } }
        if item.get("type") == "error":
            raise RuntimeError(f"Turso SQL error: {item.get('error')}")
        response = item.get("response") or {}
        if response.get("type") == "close":
            return _TursoCursor([], [], 0)
        result = response.get("result") if isinstance(response, dict) else None
        if not isinstance(result, dict):
            return _TursoCursor([], [], 0)

        cols_raw = result.get("cols") or []
        columns: list[str] = []
        for c in cols_raw:
            if isinstance(c, dict):
                columns.append(str(c.get("name") or ""))
            else:
                columns.append(str(c))

        rows_out: list[list[Any]] = []
        for row in result.get("rows") or []:
            cells = row.get("values") if isinstance(row, dict) else row
            rows_out.append([_decode_turso_value(c) for c in (cells or [])])

        affected = result.get("affected_row_count")
        if affected is None:
            affected = -1
        try:
            rowcount = int(affected)
        except (TypeError, ValueError):
            rowcount = -1
        return _TursoCursor(columns, rows_out, rowcount=rowcount)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> _TursoCursor:
        results = self._run_pipeline([self._stmt(sql, params)])
        for item in results:
            if item.get("type") == "error":
                raise RuntimeError(f"Turso SQL error: {item.get('error')}")
            resp = item.get("response") or {}
            if resp.get("type") == "execute":
                return self._parse_execute_result(item)
        return _TursoCursor([], [], 0)

    def executemany(self, sql: str, seq_of_params: Sequence[Sequence[Any]]) -> _TursoCursor:
        if not seq_of_params:
            return _TursoCursor([], [], 0)
        reqs = [self._stmt(sql, params) for params in seq_of_params]
        results = self._run_pipeline(reqs)
        total = 0
        for item in results:
            cur = self._parse_execute_result(item)
            if cur.rowcount and cur.rowcount > 0:
                total += cur.rowcount
        return _TursoCursor([], [], rowcount=total)

    def executescript(self, script: str) -> None:
        # Split on semicolons at line boundaries; skip empty / comment-only.
        parts = re.split(r";\s*\n", script)
        stmts = []
        for part in parts:
            s = part.strip()
            if not s or s.startswith("--"):
                continue
            # Drop trailing semicolon
            if s.endswith(";"):
                s = s[:-1].strip()
            if s:
                stmts.append(s)
        if not stmts:
            return
        # Batch in chunks to avoid huge payloads
        chunk_size = 8
        for i in range(0, len(stmts), chunk_size):
            batch = stmts[i : i + chunk_size]
            self._run_pipeline([self._stmt(s) for s in batch])

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


def _connect_sqlite() -> sqlite3.Connection:
    path = db_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        path = Path("/tmp/studyquiz.db")
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _connect() -> Any:
    if using_turso():
        return TursoConnection()
    return _connect_sqlite()


def _ensure_schema(conn: Any) -> None:
    """Idempotent CREATE TABLE — safe to call on every cold start."""
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        conn.executescript(_SCHEMA_SQL)
        if hasattr(conn, "commit"):
            conn.commit()
        _schema_ready = True


def reset_connection_state() -> None:
    """Drop thread-local connection and schema flag (tests / path switch)."""
    global _schema_ready
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    _local.conn = None
    _schema_ready = False


@contextmanager
def connection() -> Iterator[Any]:
    """Per-thread connection (safe with uvicorn workers / reload)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    _ensure_schema(conn)
    try:
        yield conn
        if hasattr(conn, "commit"):
            conn.commit()
    except Exception:
        if hasattr(conn, "rollback"):
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

CREATE TABLE IF NOT EXISTS exam_papers (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    use_rag INTEGER NOT NULL,
    topic TEXT,
    difficulty TEXT,
    paper_json TEXT NOT NULL,
    context_chunks_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS idx_quizzes_doc ON quizzes(document_id);
CREATE INDEX IF NOT EXISTS idx_eval_quiz ON eval_rows(quiz_id);
CREATE INDEX IF NOT EXISTS idx_eval_phase ON eval_rows(phase);
CREATE INDEX IF NOT EXISTS idx_exam_doc ON exam_papers(document_id);
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
# Exam papers (theory / Nigerian-style written exams)
# ---------------------------------------------------------------------------


def save_exam_paper(
    *,
    exam_id: str,
    document_id: str,
    use_rag: bool,
    topic: str | None,
    difficulty: str,
    paper: dict,
    context_chunks: list[str],
) -> None:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO exam_papers (
                id, document_id, use_rag, topic, difficulty,
                paper_json, context_chunks_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exam_id,
                document_id,
                int(use_rag),
                topic,
                difficulty,
                _dumps(paper),
                _dumps(context_chunks),
                _now(),
            ),
        )


def get_exam_paper(exam_id: str) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM exam_papers WHERE id = ?",
            (exam_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "doc_id": row["document_id"],
        "use_rag": bool(row["use_rag"]),
        "topic": row["topic"],
        "difficulty": row["difficulty"],
        "paper": _loads(row["paper_json"]),
        "context_chunks": _loads(row["context_chunks_json"]),
        "created_at": row["created_at"],
    }


def update_exam_paper(exam_id: str, paper: dict) -> None:
    """Overwrite the stored paper JSON (e.g. after filling answer guides)."""
    with connection() as conn:
        cur = conn.execute(
            "UPDATE exam_papers SET paper_json = ? WHERE id = ?",
            (_dumps(paper), exam_id),
        )
        # Prefer affected-row count when the backend reports it reliably.
        rc = getattr(cur, "rowcount", None)
        if rc == 0:
            exists = conn.execute(
                "SELECT 1 FROM exam_papers WHERE id = ?",
                (exam_id,),
            ).fetchone()
            if exists is None:
                raise KeyError(exam_id)


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


def _eval_row_to_dict(row: Any) -> dict:
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
