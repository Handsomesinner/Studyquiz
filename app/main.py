"""StudyQuiz — Retrieval-Augmented Question Generation from lecture materials.

FastAPI application tying the pipeline together:
  upload → extract & chunk (pdf_processor) → index (retriever)
  → generate quiz (generator, Claude) → ground-check (grounding)
  → take quiz → server-side grading.

Documents, quizzes, scores, and evaluation rows are stored in SQLite
(see ``app/store.py``) so they survive restarts.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import csv
import hashlib
import secrets
import time
import re
import io
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from . import (
    answer_from_notes,
    coverage,
    exam_generator,
    generator,
    grounding,
    pdf_processor,
    security,
    store,
)
from .retriever import BM25Retriever

app = FastAPI(title="StudyQuiz")

STATIC_DIR = Path(__file__).parent / "static"

# App-side ceiling (local multipart + blob-download path).
# On Vercel, multipart bodies are still capped by the platform (~4.5 MB);
# large files must use client → Vercel Blob → process-by-URL.
MAX_UPLOAD_BYTES = min(100, max(1, int(os.getenv("MAX_UPLOAD_MB", "100")))) * 1024 * 1024
# Guard extracted text so a huge PDF cannot explode SQLite / BM25 memory.
MAX_EXTRACTED_CHARS = int(os.getenv("MAX_EXTRACTED_CHARS", "5000000"))
# Direct multipart is only reliable under the Vercel body limit when live.
DIRECT_MULTIPART_SAFE_BYTES = 4 * 1024 * 1024

# Routes that burn Anthropic credits or expose research data when public.
_PROTECTED_PREFIXES = (
    "/api/documents",
    "/api/quiz",
    "/api/exam",
    "/api/answer-from-notes",
    "/api/evaluation",
    "/api/study",
    "/api/banks",
    "/api/share",
    "/api/blob",
    "/api/uploads",
    "/api/workspace",
)


def require_access_pin(request: Request) -> None:
    """Gate expensive / sensitive APIs when STUDYQUIZ_ACCESS_PIN is set."""
    if not security.auth_required():
        return
    path = request.url.path.rstrip("/")

    if request.method == "POST" and re.fullmatch(r"/api/share/[a-f0-9]{32}/redeem", path):
        return
    match = re.fullmatch(r"/api/(quiz|exam)/([a-f0-9]+)(/submit|/draft|/submit-writing)?", path)
    if match:
        kind, rid, action = match.groups()
        permitted = ((request.method == "GET" and action in (None, "/draft")) or
                     (request.method == "PUT" and action == "/draft") or
                     (request.method == "POST" and action in ("/submit", "/submit-writing")))
        if permitted and store.shared_resource_allowed(kind, rid):
            return
    pin = security.extract_pin_from_request_headers(request.headers)
    if not security.pin_matches(pin):
        raise HTTPException(
            401,
            "Access PIN required. Set header X-StudyQuiz-Pin (or Authorization: Bearer <pin>) "
            "to match STUDYQUIZ_ACCESS_PIN on the deployment.",
        )


@app.exception_handler(security.WorkspaceAccessError)
async def workspace_error(_request, _error):
    return JSONResponse({"detail": "Resource not found in this workspace."}, status_code=404)


@app.middleware("http")
async def access_pin_middleware(request: Request, call_next):
    path = request.url.path
    cookie = request.cookies.get("sq_workspace", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", cookie):
        cookie = secrets.token_urlsafe(32)
    ctx = security.workspace.set(hashlib.sha256(cookie.encode()).hexdigest())
    try:
        if request.method != "OPTIONS" and any(path == p or path.startswith(p + "/") for p in _PROTECTED_PREFIXES):
            if security.public_deployment() and len(security.access_pin() or "") < 16:
                return JSONResponse({"detail": "Configure a deployment access secret of at least 16 characters."}, status_code=503)
            try:
                require_access_pin(request)
            except HTTPException as e:
                return JSONResponse({"detail": e.detail}, status_code=e.status_code)
            if store._running_serverless() and not store.storage_is_durable():
                return JSONResponse({"detail": "Durable storage must be configured before this deployment can save work."}, status_code=503)
            # API requests must be same-site; PIN alone does not make CORS safe.
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin API requests are not permitted."}, status_code=403)
        response = await call_next(request)
        response.set_cookie("sq_workspace", cookie, max_age=31536000, httponly=True,
                            secure=security.public_deployment(), samesite="lax")
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
    finally:
        security.workspace.reset(ctx)


@app.get("/api/workspace")
def current_workspace():
    return {"workspace_id": security.workspace.get(), "storage": "browser-bound; clearing cookies removes access"}


class UploadPermit(BaseModel):
    pathname: str


@app.post("/api/blob/permit")
def issue_upload_permit(body: UploadPermit):
    if not re.fullmatch(r"uploads/[a-f0-9]{32}-[A-Za-z0-9_.()+-]{1,180}", body.pathname):
        raise HTTPException(400, "Invalid upload pathname.")
    day = datetime.now(timezone.utc).date().isoformat()
    owner = security.workspace.get()
    for bucket, limit in ((f"upload:global:{day}", 100), (f"upload:{owner}:{day}", 20)):
        if not store.consume_budget(bucket, limit):
            raise HTTPException(429, "Daily upload limit reached. Try again tomorrow.")
    cleanup_uploads(limit=1)  # opportunistic bounded retry; does not delay a grant by a whole queue
    with store.connection() as conn:
        conn.execute("INSERT INTO upload_permits VALUES (?, ?, ?)", (body.pathname, owner, datetime.now(timezone.utc).isoformat()))
    return {"ok": True}


def _require_upload_permit(url: str) -> str:
    # Even a valid Blob URL is not authorization: verify this store and workspace.
    if not security.is_blob_url_for_auth_header(url):
        raise HTTPException(400, "Only files uploaded to this application's private store are accepted.")
    pathname = unquote(urlparse(url).path).lstrip("/")
    with store.connection() as conn:
        row = conn.execute("SELECT workspace FROM upload_permits WHERE pathname = ?", (pathname,)).fetchone()
    if row is None or row["workspace"] != security.workspace.get():
        raise HTTPException(404, "Upload not found in this workspace.")
    return pathname


def _delete_uploaded_source(pathname: str) -> bool:
    # Vercel Blob SDK delete protocol. Retain failed cleanup records for retry.
    try:
        response = httpx.post("https://vercel.com/api/blob/delete", json={"urls": [pathname]},
                              headers={"Authorization": "Bearer " + security.blob_token(), "x-api-version": "13"}, timeout=5)
        response.raise_for_status()
    except Exception:
        return False
    with store.connection() as conn:
        conn.execute("DELETE FROM upload_permits WHERE pathname = ?", (pathname,))
        conn.execute("DELETE FROM document_sources WHERE pathname = ?", (pathname,))
    return True


@app.post("/api/uploads/cleanup")
def cleanup_uploads(limit: int = 10):
    limit = max(1, min(limit, 10))
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with store.connection() as conn:
        rows = conn.execute("SELECT pathname FROM upload_permits WHERE workspace = ? AND created_at < ? LIMIT ?",
                            (security.workspace.get(), cutoff, limit)).fetchall()
    deleted = sum(_delete_uploaded_source(r["pathname"]) for r in rows)
    return {"deleted": deleted, "retry_pending": len(rows) - deleted}


@app.on_event("startup")
def _startup() -> None:
    # Best-effort: Vercel cold starts may skip or re-run this. Schema is also
    # applied lazily on the first DB use so a startup failure cannot 500 the site.
    try:
        store.init_db()
    except (OSError, RuntimeError):
        # Keep the UI/health page available to explain missing configuration;
        # protected APIs still fail closed until durable storage is configured.
        pass


@app.get("/")
def home():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(500, "UI file missing from deployment bundle.")
    return FileResponse(index)


@app.get("/api/health")
def health():
    """Lightweight check used to confirm the serverless function boots."""
    info = store.storage_info()
    try:
        store.init_db()
        ok = True
        detail = "ok"
    except Exception:
        ok = False
        detail = "Storage unavailable; check deployment configuration."
    return {
        "status": "ok" if ok else "degraded",
        "workspace_id": security.workspace.get(),
        "storage": {"durable": info["durable"], "hint": info["hint"]},
        "storage_backend": info["backend"],
        "storage_durable": info["durable"],
        "serverless": store._running_serverless(),
        "detail": detail,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "blob_configured": bool(security.blob_token()),
        "blob_access": (
            "private"
            if (os.getenv("BLOB_ACCESS") or "private").strip().lower() == "private"
            else "public"
        ),
        "auth_required": security.auth_required(),
        "upload_hint": (
            "Use Vercel Blob client upload for files over ~4 MB on Vercel. "
            "Scanned PDFs need ANTHROPIC_API_KEY for OCR after upload."
            if store._running_serverless()
            else "Direct multipart upload is fine locally."
        ),
    }


def _index_document_bytes(
    filename: str,
    data: bytes,
    *,
    replace_doc_id: str | None = None,
) -> dict:
    """Extract, chunk, and persist a document from raw bytes.

    When ``replace_doc_id`` is set, re-indexes that existing document in place
    (same id) instead of creating a new row.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    ocr_used = False
    try:
        text, ocr_used = pdf_processor.extract_text_with_meta(
            filename or "upload.pdf", data
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(
            400,
            "Could not read this file. Make sure it is a valid, non-corrupted "
            "document (PDF, Word, PowerPoint, or a text-based file).",
        )

    if len(text) > MAX_EXTRACTED_CHARS:
        raise HTTPException(
            413,
            "Extracted text is too large to index in this demo deployment. "
            "Try a shorter document or split the lecture pack.",
        )

    chunks = pdf_processor.chunk_text(text)
    if len(chunks) == 0:
        raise HTTPException(
            400,
            "No text could be extracted from this document. "
            "If it is a scan, ensure ANTHROPIC_API_KEY is set (OCR fallback) "
            "and the pages are readable.",
        )

    title = filename or "untitled"
    if replace_doc_id:
        if not store.replace_document(
            doc_id=replace_doc_id,
            title=title,
            text=text,
            chunks=chunks,
        ):
            raise HTTPException(404, "Document not found. Upload it again.")
        doc_id = replace_doc_id
        replaced = True
    else:
        doc_id = uuid.uuid4().hex[:12]
        store.save_document(
            doc_id=doc_id,
            title=title,
            text=text,
            chunks=chunks,
        )
        replaced = False
    if security.source_path.get():
        with store.connection() as conn:
            conn.execute("INSERT OR REPLACE INTO document_sources VALUES (?, ?)", (doc_id, security.source_path.get()))
    return {
        "id": doc_id,
        "title": title,
        "num_chunks": len(chunks),
        "num_words": len(text.split()),
        "ocr_used": ocr_used,
        "replaced": replaced,
    }


def _filename_from_url(url: str, fallback: str = "upload.bin") -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name
    return name or fallback


def _download_url_to_bytes(url: str) -> bytes:
    """Download a remote file (Vercel Blob only) with SSRF allowlist + size ceiling.

    Only https hosts under the Blob allowlist are fetched. Redirects are followed
    manually and re-checked against the same allowlist.
    """
    try:
        url = security.validate_download_url(url)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    _require_upload_permit(url)
    max_bytes = MAX_UPLOAD_BYTES
    headers: dict[str, str] = {}
    blob_token = security.blob_token()
    if blob_token and security.is_blob_url_for_auth_header(url):
        headers["Authorization"] = f"Bearer {blob_token}"

    current = url
    try:
        with httpx.Client(follow_redirects=False, timeout=180.0, headers=headers) as client:
            for _ in range(5):
                with client.stream("GET", current) as resp:
                    # Manual redirect follow with allowlist re-check (SSRF).
                    if resp.status_code in (301, 302, 303, 307, 308):
                        loc = resp.headers.get("location")
                        if not loc:
                            raise HTTPException(400, "Redirect without Location header.")
                        # Resolve relative redirects against current URL.
                        next_url = str(httpx.URL(current).join(loc))
                        try:
                            current = security.validate_download_url(next_url)
                        except ValueError as e:
                            raise HTTPException(
                                400,
                                f"Redirect target blocked: {e}",
                            ) from e
                        if urlparse(current).hostname != urlparse(url).hostname:
                            raise HTTPException(400, "Cross-store redirects are not permitted.")
                        continue

                    if resp.status_code >= 400:
                        raise HTTPException(
                            400,
                            f"Could not download uploaded file (HTTP {resp.status_code}). "
                            "If the Blob store is Private, ensure BLOB_READ_WRITE_TOKEN is set.",
                        )
                    cl = resp.headers.get("content-length")
                    if cl and cl.isdigit() and int(cl) > max_bytes:
                        raise HTTPException(
                            413,
                            f"File too large (max {max_bytes // (1024 * 1024)} MB).",
                        )
                    buf = bytearray()
                    for chunk in resp.iter_bytes(1024 * 256):
                        buf.extend(chunk)
                        if len(buf) > max_bytes:
                            raise HTTPException(
                                413,
                                f"File too large (max {max_bytes // (1024 * 1024)} MB).",
                            )
                    return bytes(buf)
            raise HTTPException(400, "Too many redirects while downloading file.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            400,
            "Failed to download the uploaded file. Please retry.",
        ) from e


@contextmanager
def uploaded_source(url: str):
    pathname = _require_upload_permit(url)
    source_ctx = security.source_path.set(pathname)
    try:
        yield _download_url_to_bytes(url)
    finally:
        security.source_path.reset(source_ctx)
        if not _delete_uploaded_source(pathname):
            # No URL/token in logs. Cleanup queue survives worker restarts.
            import logging
            logging.getLogger(__name__).warning("Uploaded source cleanup pending")


async def _read_bounded_file(file) -> bytes:
    ceiling = DIRECT_MULTIPART_SAFE_BYTES if store._running_serverless() else MAX_UPLOAD_BYTES
    data = await file.read(ceiling + 1)
    if len(data) > ceiling:
        raise HTTPException(413, "File exceeds the upload size limit. Use a smaller file or private Blob upload.")
    return data


class DocumentFromUrl(BaseModel):
    """Index a file already uploaded to object storage (Vercel Blob, etc.)."""

    url: str = Field(..., description="Public or readable blob URL")
    filename: str | None = None


@app.post("/api/documents")
async def upload_document(request: Request):
    """Accept either multipart file upload or JSON {url, filename}.

    Large files on Vercel must use the Blob client path (JSON url), because
    serverless request bodies are capped around 4.5 MB.
    """
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            payload = DocumentFromUrl.model_validate(await request.json())
        except Exception:
            raise HTTPException(
                400,
                "JSON body must include a string 'url' (and optional 'filename').",
            )
        with uploaded_source(payload.url) as data:
            filename = payload.filename or _filename_from_url(payload.url)
            return _index_document_bytes(filename, data)

    # Classic multipart (local dev / small files).
    form = await request.form()
    file = form.get("file")
    if file is None or not hasattr(file, "read"):
        raise HTTPException(
            400,
            "Expected multipart field 'file', or JSON {url, filename} after "
            "Vercel Blob client upload.",
        )
    data = await _read_bounded_file(file)
    filename = getattr(file, "filename", None) or "upload.bin"
    if (
        store._running_serverless()
        and len(data) > DIRECT_MULTIPART_SAFE_BYTES
    ):
        raise HTTPException(
            413,
            "On Vercel, files over ~4 MB must use Blob client upload "
            "(the UI does this automatically when BLOB_READ_WRITE_TOKEN is set).",
        )
    return _index_document_bytes(filename, data)


@app.get("/api/documents")
def list_documents():
    return store.list_documents()



class DocumentRename(BaseModel):
    title: str


@app.patch("/api/documents/{doc_id}")
def rename_document(doc_id: str, body: DocumentRename):
    """Rename a saved document (display title only)."""
    title = (body.title or "").strip()
    if not title:
        raise HTTPException(400, "Title is required.")
    if len(title) > 300:
        raise HTTPException(400, "Title is too long (max 300 characters).")
    if not store.rename_document(doc_id, title):
        raise HTTPException(404, "Document not found.")
    return {"id": doc_id, "title": title}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete the source file first, then related saved work."""
    store.document_exists(doc_id)  # ownership guard before any storage action
    with store.connection() as conn:
        sources = conn.execute("SELECT pathname FROM document_sources WHERE document_id = ?", (doc_id,)).fetchall()
    for source in sources:
        if not _delete_uploaded_source(source["pathname"]):
            raise HTTPException(503, "Original-file deletion could not be confirmed. Your saved work was retained so you can retry deletion.")
    if not store.delete_document(doc_id):
        raise HTTPException(404, "Document not found.")
    return {"ok": True, "id": doc_id}


@app.put("/api/documents/{doc_id}")
async def replace_document(doc_id: str, request: Request):
    """Replace file content for an existing document (same id, re-chunked).

    Accepts the same body styles as POST /api/documents:
    multipart ``file`` or JSON ``{url, filename}`` after Blob upload.
    """
    if not store.document_exists(doc_id):
        raise HTTPException(404, "Document not found. Upload it again.")

    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            payload = DocumentFromUrl.model_validate(await request.json())
        except Exception:
            raise HTTPException(
                400,
                "JSON body must include a string 'url' (and optional 'filename').",
            )
        with uploaded_source(payload.url) as data:
            filename = payload.filename or _filename_from_url(payload.url)
            return _index_document_bytes(filename, data, replace_doc_id=doc_id)

    form = await request.form()
    file = form.get("file")
    if file is None or not hasattr(file, "read"):
        raise HTTPException(
            400,
            "Expected multipart field 'file', or JSON {url, filename} after "
            "Vercel Blob client upload.",
        )
    data = await _read_bounded_file(file)
    filename = getattr(file, "filename", None) or "upload.bin"
    if (
        store._running_serverless()
        and len(data) > DIRECT_MULTIPART_SAFE_BYTES
    ):
        raise HTTPException(
            413,
            "On Vercel, files over ~4 MB must use Blob client upload "
            "(the UI does this automatically when BLOB_READ_WRITE_TOKEN is set).",
        )
    return _index_document_bytes(filename, data, replace_doc_id=doc_id)


class QuizRequest(BaseModel):
    document_id: str
    num_questions: int = 5
    topic: str | None = None
    use_rag: bool = True  # False = ungrounded baseline for the evaluation study
    # When True (default) and use_rag=True, drop questions whose source_quote
    # is not found in the document so students only see verified items.
    # Evaluation metrics still record the full pre-filter generation.
    require_grounding: bool = True
    difficulty: str = "medium"  # easy | medium | hard
    harder_distractors: bool = False  # closer wrong options


def _build_eval_rows(
    *,
    quiz_id: str,
    doc_id: str,
    doc_title: str,
    topic: str | None,
    use_rag: bool,
    questions: list,
    groundings: list,
    phase: str,
) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, (q, g) in enumerate(zip(questions, groundings)):
        rows.append(
            {
                "timestamp": now,
                "quiz_id": quiz_id,
                "document_id": doc_id,
                "document_title": doc_title,
                "use_rag": use_rag,
                "topic": topic or "",
                "phase": phase,
                "question_index": i,
                "question": q.question,
                "options": " | ".join(q.options),
                "correct_index": q.correct_index,
                "source_quote": q.source_quote or "",
                "grounded": g.grounded,
                "match_type": g.match_type,
                "options_unique": g.options_unique,
                "expected_grounded": g.expected_grounded,
            }
        )
    return rows


def _generate_mcq_attempt(
    *,
    doc: dict,
    num_questions: int,
    plan,
    context_batches: list,
    topic: str | None,
    use_rag: bool,
    difficulty: str,
    source_text: str,
    harder_distractors: bool = False,
):
    """One generation + grounding pass. Raises GenerationError on API failure."""
    quiz = generator.generate_quiz_from_batches(
        num_questions=num_questions,
        doc_title=doc["title"],
        context_batches=context_batches,
        questions_per_batch=plan.questions_per_batch,
        topic=topic,
        use_rag=use_rag,
        difficulty=difficulty,
        harder_distractors=harder_distractors,
    )
    pre_groundings, pre_metrics = grounding.validate_quiz(
        quiz.questions,
        source_text=source_text,
        use_rag=use_rag,
    )
    return quiz, pre_groundings, pre_metrics


@app.post("/api/quiz")
def create_quiz(req: QuizRequest):
    from . import quality

    doc = store.get_document(req.document_id)
    if doc is None:
        raise HTTPException(404, "Document not found. Upload it again.")
    num_questions = max(1, min(req.num_questions, 15))

    # Retrieval plan: scale top_k with doc size / num_questions, diversify
    # topic hits, and batch long selections for section-wise generation.
    retriever: BM25Retriever = doc["retriever"]
    plan = coverage.plan_retrieval(
        retriever,
        num_questions=num_questions,
        topic=req.topic,
        use_rag=req.use_rag,
    )
    context_batches = coverage.context_batches_from_plan(plan, doc["chunks"])
    # Flat list still stored for evaluation / debugging.
    context_chunks = [doc["chunks"][i] for i in plan.chunk_indices] if plan.chunk_indices else []
    source_text = doc.get("text") or "\n\n".join(doc["chunks"])

    run_id = uuid.uuid4().hex
    raw_attempts = []
    started = time.monotonic()
    def generate_attempt(**kwargs):
        try:
            result = _generate_mcq_attempt(**kwargs)
        except Exception as error:
            store.record_generation_run(run_id, req.document_id, req.use_rag, "attempt_failed",
                                        {"attempts": raw_attempts, "elapsed_seconds": time.monotonic() - started,
                                         "error_type": type(error).__name__, "http_status": getattr(error, "status_code", None)})
            raise
        raw_attempts.append({"questions": [q.model_dump() for q in result[0].questions], "metrics": result[2].to_dict()})
        store.record_generation_run(run_id, req.document_id, req.use_rag, "generated", {"attempts": raw_attempts})
        return result

    try:
        quiz, pre_groundings, pre_metrics = generate_attempt(
            doc=doc,
            num_questions=num_questions,
            plan=plan,
            context_batches=context_batches,
            topic=req.topic,
            use_rag=req.use_rag,
            difficulty=req.difficulty,
            source_text=source_text,
            harder_distractors=req.harder_distractors,
        )
    except generator.GenerationError as e:
        store.record_generation_run(run_id, req.document_id, req.use_rag, "generation_failed", {"attempts": raw_attempts, "elapsed_seconds": time.monotonic() - started})
        raise HTTPException(503, str(e))

    original_questions = list(quiz.questions)
    original_groundings = list(pre_groundings)
    original_metrics = pre_metrics
    if not quiz.questions:
        raise HTTPException(
            503,
            "The model returned no questions. Try again or use a longer document.",
        )

    # Near-duplicate filter on full model output before grounding serve policy.
    deduped_q, deduped_g, duplicates_dropped = quality.filter_near_duplicates(
        quiz.questions, pre_groundings
    )
    if deduped_q:
        quiz.questions = deduped_q
        if deduped_g is not None:
            pre_groundings = deduped_g
            pre_metrics = grounding.summarise(pre_groundings, use_rag=req.use_rag)

    # Retry only ungrounded items (not the whole quiz) when require_grounding.
    partial_retries = 0
    if req.use_rag and req.require_grounding:
        bad_idx = quality.ungrounded_indices(pre_groundings, use_rag=True)
        if bad_idx and len(bad_idx) < len(quiz.questions):
            n_retry = len(bad_idx)
            try:
                repl_quiz, repl_g, _ = generate_attempt(
                    doc=doc,
                    num_questions=n_retry,
                    plan=plan,
                    context_batches=context_batches,
                    topic=req.topic,
                    use_rag=req.use_rag,
                    difficulty=req.difficulty,
                    source_text=source_text,
                    harder_distractors=req.harder_distractors,
                )
                # Slot replacements into original positions when grounded
                ri = 0
                for bi in bad_idx:
                    if ri >= len(repl_quiz.questions):
                        break
                    cand = repl_quiz.questions[ri]
                    cg = repl_g[ri]
                    ri += 1
                    # Skip near-dup of any kept question
                    if any(
                        quality.is_near_duplicate(cand, kq)
                        for j, kq in enumerate(quiz.questions)
                        if j != bi
                    ):
                        continue
                    if cg.grounded:
                        quiz.questions[bi] = cand
                        pre_groundings[bi] = cg
                        partial_retries += 1
                pre_metrics = grounding.summarise(
                    pre_groundings, use_rag=req.use_rag
                )
            except generator.GenerationError:
                pass
        elif bad_idx and len(bad_idx) == len(quiz.questions):
            # All ungrounded: one full regenerate (existing soft-fallback path)
            try:
                quiz2, pre2, metrics2 = generate_attempt(
                    doc=doc,
                    num_questions=num_questions,
                    plan=plan,
                    context_batches=context_batches,
                    topic=req.topic,
                    use_rag=req.use_rag,
                    difficulty=req.difficulty,
                    source_text=source_text,
                    harder_distractors=req.harder_distractors,
                )
                if quiz2.questions:
                    dq, dg, dd = quality.filter_near_duplicates(
                        quiz2.questions, pre2
                    )
                    if dq:
                        quiz.questions = dq
                        pre_groundings = dg if dg is not None else pre2
                        pre_metrics = metrics2
                        duplicates_dropped += dd
                        partial_retries = len(dq)
            except generator.GenerationError:
                pass

    served_questions, served_groundings, filtered_out, best_effort = (
        grounding.resolve_grounded_serving(
            quiz.questions,
            pre_groundings,
            use_rag=req.use_rag,
            require_grounding=req.require_grounding,
        )
    )

    # Strict mode never silently relaxes verification.
    retried = partial_retries > 0

    post_metrics = grounding.summarise(served_groundings, use_rag=req.use_rag)

    warning = None
    if best_effort:
        total = len(served_questions)
        grounded_n = sum(1 for g in served_groundings if g.grounded)
        warning = (
            "Could not verify source quotes for these questions"
            f" ({grounded_n}/{total} matched the notes)"
            + (" after retrying ungrounded items. " if retried else ". ")
            + "Serving best-effort questions — answers may be less tightly grounded. "
            "You can generate again or turn off “Only keep questions verified against the notes”."
        )
    elif duplicates_dropped or partial_retries:
        bits = []
        if duplicates_dropped:
            bits.append(f"removed {duplicates_dropped} near-duplicate(s)")
        if partial_retries:
            bits.append(f"replaced {partial_retries} ungrounded item(s)")
        warning = "Quality pass: " + "; ".join(bits) + "."

    quiz_id = uuid.uuid4().hex[:12]
    store.save_quiz(
        quiz_id=quiz_id,
        document_id=req.document_id,
        use_rag=req.use_rag,
        topic=req.topic,
        questions=served_questions,
        groundings=served_groundings,
        pre_filter_metrics=original_metrics.to_dict(),
        served_metrics=post_metrics.to_dict(),
        context_chunks=context_chunks,
    )

    store.append_eval_rows(
        _build_eval_rows(
            quiz_id=quiz_id,
            doc_id=req.document_id,
            doc_title=doc["title"] or "",
            topic=req.topic,
            use_rag=req.use_rag,
            questions=original_questions,
            groundings=original_groundings,
            phase="pre_filter",
        )
        + _build_eval_rows(
            quiz_id=quiz_id,
            doc_id=req.document_id,
            doc_title=doc["title"] or "",
            topic=req.topic,
            use_rag=req.use_rag,
            questions=served_questions,
            groundings=served_groundings,
            phase="served",
        )
    )

    store.record_generation_run(run_id, req.document_id, req.use_rag,
                                "served" if served_questions else "verification_failed",
                                {"attempts": raw_attempts, "served_count": len(served_questions),
                                 "elapsed_seconds": time.monotonic() - started,
                                 "accuracy_measured": False})
    if not served_questions:
        raise HTTPException(422, "No questions passed the source-quote and option checks. No unverified quiz was served. Try different notes or explicitly turn off the quote filter.")

    # Answers stay server-side; the client only sees questions and options.
    # Grounding flags are revealed after submit (or via evaluation endpoints).
    return {
        "quiz_id": quiz_id,
        "use_rag": req.use_rag,
        "difficulty": generator._normalize_difficulty(req.difficulty),
        "questions": [
            {"index": i, "question": q.question, "options": q.options}
            for i, q in enumerate(served_questions)
        ],
        "retrieval": plan.to_dict(),
        "grounding": {
            "require_grounding": req.require_grounding and req.use_rag,
            "filtered_out": filtered_out,
            "pre_filter": original_metrics.to_dict(),
            "served": post_metrics.to_dict(),
            "retried": retried,
            "partial_retries": partial_retries,
            "duplicates_dropped": duplicates_dropped,
            "best_effort": best_effort,
            "warning": warning,
            "harder_distractors": bool(req.harder_distractors),
        },
    }


class SubmitRequest(BaseModel):
    answers: list[int]  # chosen option index per question, -1 = unanswered


@app.post("/api/quiz/{quiz_id}/submit")
def submit_quiz(quiz_id: str, req: SubmitRequest):
    quiz = store.get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz not found.")
    questions = quiz["questions"]
    groundings = quiz.get("groundings") or []
    if len(req.answers) != len(questions):
        raise HTTPException(400, "Answer count does not match question count.")

    results = []
    score = 0
    for i, (q, chosen) in enumerate(zip(questions, req.answers)):
        correct = chosen == q.correct_index
        score += int(correct)
        g = groundings[i] if i < len(groundings) else None
        results.append(
            {
                "correct": correct,
                "chosen_index": chosen,
                "correct_index": q.correct_index,
                "explanation": q.explanation,
                "source_quote": q.source_quote,
                "grounded": g.grounded if g else False,
                "match_type": g.match_type if g else "unknown",
            }
        )

    store.save_attempt(
        attempt_id=uuid.uuid4().hex[:12],
        quiz_id=quiz_id,
        answers=req.answers,
        score=score,
        total=len(questions),
        results=results,
    )

    # Feed spaced-repetition bank (wrong → due now; known cards step on correct)
    srs_updated = 0
    try:
        from . import study as study_mod

        doc_id = quiz.get("doc_id")
        if doc_id and store.resource_allowed("doc", doc_id):
            for i, (q, r) in enumerate(zip(questions, results)):
                fp = study_mod.fingerprint_question(q.question)
                payload = {
                    "question": q.question,
                    "options": list(q.options),
                    "correct_index": q.correct_index,
                    "explanation": q.explanation or "",
                    "source_quote": q.source_quote or "",
                }
                existing = store.get_srs_card_by_fingerprint(doc_id, fp)
                if not r.get("correct"):
                    store.upsert_srs_card(
                        card_id=(existing["id"] if existing else uuid.uuid4().hex[:12]),
                        document_id=doc_id,
                        fingerprint=fp,
                        card_type="mcq",
                        payload=payload,
                        ease=existing["ease"] if existing else 2.5,
                        interval_days=0,
                        due_at=study_mod.utc_now().isoformat(),
                        reps=0,
                        lapses=(existing["lapses"] + 1) if existing else 1,
                    )
                    srs_updated += 1
                elif existing:
                    sched = study_mod.sm2_schedule(
                        grade="good",
                        ease=existing["ease"],
                        interval_days=existing["interval_days"],
                        reps=existing["reps"],
                        lapses=existing["lapses"],
                    )
                    store.update_srs_card_schedule(
                        existing["id"],
                        ease=sched["ease"],
                        interval_days=sched["interval_days"],
                        due_at=sched["due_at"],
                        reps=sched["reps"],
                        lapses=sched["lapses"],
                    )
                    srs_updated += 1
    except Exception:
        srs_updated = 0

    return {
        "score": score,
        "total": len(questions),
        "results": results,
        "grounding": quiz.get("served_metrics"),
        "srs_updated": srs_updated,
    }


@app.get("/api/quiz/{quiz_id}/evaluation")
def quiz_evaluation(quiz_id: str):
    if not store.resource_allowed("quiz", quiz_id):
        raise security.WorkspaceAccessError()
    """JSON evaluation summary for one quiz (for the report / tooling)."""
    quiz = store.get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz not found.")
    doc = store.get_document(quiz["doc_id"])
    return {
        "quiz_id": quiz_id,
        "document_id": quiz["doc_id"],
        "document_title": doc.get("title") if doc else None,
        "use_rag": quiz["use_rag"],
        "topic": quiz.get("topic"),
        "pre_filter_metrics": quiz.get("pre_filter_metrics"),
        "served_metrics": quiz.get("served_metrics"),
        "questions": [
            {
                "index": i,
                "question": q.question,
                "options": q.options,
                "correct_index": q.correct_index,
                "source_quote": q.source_quote,
                "grounding": g.to_dict(),
            }
            for i, (q, g) in enumerate(
                zip(quiz["questions"], quiz.get("groundings") or [])
            )
        ],
    }


class QuestionReview(BaseModel):
    answer_correct: bool
    evidence_supports: bool
    clear_question: bool
    notes: str = Field(default="", max_length=2000)


@app.put("/api/quiz/{quiz_id}/reviews/{question_index}")
def review_question(quiz_id: str, question_index: int, body: QuestionReview):
    if not store.resource_allowed("quiz", quiz_id):
        raise security.WorkspaceAccessError()
    quiz = store.get_quiz(quiz_id)
    if quiz is None or not 0 <= question_index < len(quiz["questions"]):
        raise HTTPException(404, "Question not found.")
    store.save_question_review(quiz_id, question_index, **body.model_dump())
    return {"ok": True, "method": "human review; not automatic accuracy"}


@app.get("/api/evaluation/export")
def export_evaluation_csv():
    """Download all recorded question-level evaluation rows as CSV.

    Columns support the evaluation chapter: compare RAG vs baseline on
    quote_in_source (grounded), match_type, and options_unique.
    """
    eval_rows = store.list_eval_rows()
    if not eval_rows:
        raise HTTPException(
            404,
            "No evaluation data yet. Generate at least one quiz first.",
        )

    reviews = {(r["quiz_id"], r["question_index"]): r for r in store.list_question_reviews()}
    fieldnames = [
        "timestamp",
        "quiz_id",
        "document_id",
        "document_title",
        "use_rag",
        "topic",
        "phase",
        "question_index",
        "question",
        "options",
        "correct_index",
        "source_quote",
        "grounded",
        "match_type",
        "options_unique",
        "expected_grounded",
        "answer_correct", "evidence_supports", "clear_question", "review_notes",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in eval_rows:
        review = reviews.get((row.get("quiz_id"), row.get("question_index")), {}) if row.get("phase") == "served" else {}
        row = {**row, **{k: review.get(k, "") for k in ("answer_correct", "evidence_supports", "clear_question")}, "review_notes": review.get("notes", "")}
        writer.writerow({k: ("'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else v) for k, v in row.items()})

    buf.seek(0)
    filename = f"studyquiz_evaluation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/evaluation/summary")
def evaluation_summary():
    """Aggregate quote-in-source rates by condition (RAG vs baseline)."""
    eval_rows = store.list_eval_rows()

    # Use pre_filter rows so filtering does not hide model failure rate.
    rows = [r for r in eval_rows if r.get("phase") == "pre_filter"]
    if not rows:
        rows = eval_rows

    by: dict[str, dict] = {}
    for r in rows:
        key = "rag" if r.get("use_rag") else "baseline"
        bucket = by.setdefault(
            key,
            {"questions": 0, "grounded": 0, "options_unique": 0, "quizzes": set()},
        )
        bucket["questions"] += 1
        bucket["grounded"] += int(bool(r.get("grounded")))
        bucket["options_unique"] += int(bool(r.get("options_unique")))
        bucket["quizzes"].add(r.get("quiz_id"))

    out = {}
    for key, bucket in by.items():
        n = bucket["questions"] or 1
        out[key] = {
            "quiz_count": len(bucket["quizzes"]),
            "question_count": bucket["questions"],
            "quote_in_source_rate": bucket["grounded"] / n,
            "options_unique_rate": bucket["options_unique"] / n,
        }
    reviewed = store.list_question_reviews()
    human = {}
    for condition in (True, False):
        items = [r for r in reviewed if r["use_rag"] == condition]
        human["rag" if condition else "baseline"] = {
            "reviewed_questions": len(items),
            "answer_correct_rate": sum(r["answer_correct"] for r in items) / len(items) if items else None,
            "evidence_support_rate": sum(r["evidence_supports"] for r in items) / len(items) if items else None,
            "clear_question_rate": sum(r["clear_question"] for r in items) / len(items) if items else None,
        }
    return {"quizzes": len({r.get("quiz_id") for r in eval_rows}), "by_condition": out, "human_review": human, "metric": "quote provenance, not answer correctness",
            "accuracy_comparison_available": False,
            "note": "Baseline prompts do not request source quotes. Do not interpret quote rates as comparative accuracy.",
            "generation_runs": store.list_generation_runs()}


# ---------------------------------------------------------------------------
# Exam Quiz — Nigerian-style theory / written examination papers
# ---------------------------------------------------------------------------


class ExamRequest(BaseModel):
    document_id: str
    num_questions: int = 3  # major questions (QUESTION ONE …); default 3 for speed
    topic: str | None = None  # optional focus; when set, steers retrieval + prompts
    use_rag: bool = True
    difficulty: str = "medium"
    course_code: str | None = None
    course_title: str | None = None
    time_allowed: str = "2 Hrs."


@app.post("/api/exam")
def create_exam(req: ExamRequest):
    """Generate a theory exam paper (parallel per-question calls for speed)."""
    doc = store.get_document(req.document_id)
    if doc is None:
        raise HTTPException(404, "Document not found. Upload it again.")

    topic = (req.topic or "").strip() or None
    num_questions = max(2, min(req.num_questions, 6))
    all_chunks: list[str] = list(doc["chunks"])

    # Retrieval metadata for the UI (topic-focused when a topic is given).
    retriever: BM25Retriever = doc["retriever"]
    plan = coverage.plan_retrieval(
        retriever,
        num_questions=num_questions,
        topic=topic,
        use_rag=req.use_rag,
    )

    try:
        paper, gen_meta = exam_generator.generate_exam_paper(
            num_questions=num_questions,
            doc_title=doc["title"] or "Lecture material",
            context_chunks=all_chunks,
            all_document_chunks=all_chunks,
            topic=topic,
            use_rag=req.use_rag,
            difficulty=req.difficulty,
            course_code=req.course_code,
            course_title=req.course_title,
            time_allowed=req.time_allowed or "2 Hrs.",
            retriever=retriever,
        )
    except exam_generator.GenerationError as e:
        raise HTTPException(503, str(e))

    exam_id = uuid.uuid4().hex[:12]
    paper_dict = paper.model_dump()
    # Soft quote badges on every part (never drop exam content).
    source_text = doc.get("text") or "\n\n".join(all_chunks)
    paper_dict = grounding.annotate_exam_paper_dict(paper_dict, source_text)
    # Store a compact sample of context used (first slice) for debugging only.
    sample_ctx = exam_generator.assign_chunk_slices(all_chunks, 1)[0] if all_chunks else []
    store.save_exam_paper(
        exam_id=exam_id,
        document_id=req.document_id,
        use_rag=req.use_rag,
        topic=topic,
        difficulty=generator._normalize_difficulty(req.difficulty),
        paper=paper_dict,
        context_chunks=sample_ctx,
    )

    total_marks = exam_generator.paper_total_marks(paper)
    strategy = (
        "parallel_per_question_topic_focused" if topic else "parallel_per_question"
    )
    return {
        "exam_id": exam_id,
        "mode": "exam",
        "difficulty": generator._normalize_difficulty(req.difficulty),
        "use_rag": req.use_rag,
        "topic": topic,
        "total_marks": total_marks,
        "retrieval": {
            **plan.to_dict(),
            "exam_strategy": strategy,
            "focus_topic": topic,
            "chunks_per_question": exam_generator.MAX_CHUNKS_PER_QUESTION,
            "questions_requested": gen_meta.get("requested", num_questions),
            "questions_generated": len(paper.questions),
            "partial": gen_meta.get("partial", False),
        },
        "generation": gen_meta,
        "paper": paper_dict,
        "grounding": paper_dict.get("grounding_summary"),
        "guides_deferred": True,
        "warning": gen_meta.get("message"),
    }


@app.get("/api/exam/{exam_id}")
def get_exam(exam_id: str):
    row = store.get_exam_paper(exam_id)
    if row is None:
        raise HTTPException(404, "Exam paper not found.")
    paper = row["paper"]
    # Recompute total if needed
    total = 0.0
    for q in paper.get("questions") or []:
        for p in q.get("parts") or []:
            total += float(p.get("marks") or 0)
    has_guides = any(
        (p.get("guide_points") or p.get("answer_outline"))
        for q in paper.get("questions") or []
        for p in q.get("parts") or []
    )
    return {
        "exam_id": exam_id,
        "mode": "exam",
        "difficulty": row["difficulty"],
        "use_rag": row["use_rag"],
        "topic": row["topic"],
        "total_marks": total,
        "paper": paper,
        "grounding": (paper or {}).get("grounding_summary"),
        "created_at": row["created_at"],
        "guides_ready": has_guides,
    }


@app.post("/api/exam/{exam_id}/answers")
def generate_exam_answers(exam_id: str):
    if not store.resource_allowed("exam", exam_id):
        raise security.WorkspaceAccessError()
    """Generate marking scheme + model-answer outlines from the lecture notes.

    Call this *after* attempting the paper on your own. Answers are grounded
    in the uploaded document (BM25 retrieval per question) so you can self-check.
    """
    row = store.get_exam_paper(exam_id)
    if row is None:
        raise HTTPException(404, "Exam paper not found. Generate a paper first.")

    doc = store.get_document(row["doc_id"])
    if doc is None:
        raise HTTPException(
            404,
            "Original document is no longer available. Re-upload the notes and "
            "generate a new exam paper.",
        )

    try:
        paper = exam_generator.ExamPaper.model_validate(row["paper"])
    except Exception:
        raise HTTPException(500, "Stored exam paper is corrupted. Generate a new one.")

    try:
        filled = exam_generator.fill_answer_guides(
            paper,
            doc_title=doc["title"] or "Lecture material",
            chunks=list(doc["chunks"]),
            retriever=doc["retriever"],
        )
    except exam_generator.GenerationError as e:
        raise HTTPException(503, str(e))

    paper_dict = filled.model_dump()
    source_text = doc.get("text") or "\n\n".join(doc["chunks"])
    paper_dict = grounding.annotate_exam_paper_dict(paper_dict, source_text)
    try:
        store.update_exam_paper(exam_id, paper_dict)
    except KeyError:
        raise HTTPException(404, "Exam paper not found.")

    total = exam_generator.paper_total_marks(filled)
    return {
        "exam_id": exam_id,
        "mode": "exam",
        "guides_ready": True,
        "total_marks": total,
        "paper": paper_dict,
        "grounding": paper_dict.get("grounding_summary"),
        "message": (
            "Marking guides and model-answer outlines are ready. "
            "Compare them with what you wrote — they are based on your uploaded notes."
        ),
    }


# ---------------------------------------------------------------------------
# My questions — user-supplied questions, answers from uploaded notes
# ---------------------------------------------------------------------------


def _questions_from_file_bytes(file_name: str, file_bytes: bytes) -> list[str]:
    """Extract and parse questions from a past-paper / questions file."""
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"Questions file too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    try:
        q_text, _ = pdf_processor.extract_text_with_meta(
            file_name or "questions.txt", file_bytes
        )
    except ValueError as e:
        raise HTTPException(400, f"Could not read questions file: {e}")
    except Exception:
        raise HTTPException(400, "Could not read the questions file.")
    return answer_from_notes.parse_questions(q_text)


@app.post("/api/answer-from-notes")
async def answer_from_notes_endpoint(request: Request):
    """Answer the student's own questions using retrieved lecture notes.

    Accepts JSON:
      {
        "document_id",
        "questions_text"?,
        "questions"?: ["..."],
        "questions_file_url"?,   # after Vercel Blob client upload (large files)
        "questions_filename"?
      }
    or multipart form:
      document_id, questions_text?, file? (questions file PDF/DOCX/txt)
    """
    content_type = (request.headers.get("content-type") or "").lower()
    document_id = ""
    questions_text = ""
    file_bytes: bytes | None = None
    file_name = ""
    pre_parsed: list[str] = []
    is_json = "application/json" in content_type

    bank_id = ""
    if is_json:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON body.")
        document_id = str(body.get("document_id") or "").strip()
        questions_text = str(body.get("questions_text") or body.get("text") or "")
        bank_id = str(body.get("bank_id") or "").strip()
        raw_list = body.get("questions")
        if isinstance(raw_list, list):
            pre_parsed = [str(q).strip() for q in raw_list if str(q).strip()]
        # Large questions files: browser → Vercel Blob → process by URL
        # (same path as lecture notes; avoids ~4.5 MB serverless body limit).
        file_url = str(
            body.get("questions_file_url") or body.get("file_url") or ""
        ).strip()
        if file_url:
            file_name = (
                str(body.get("questions_filename") or body.get("filename") or "").strip()
                or _filename_from_url(file_url, "questions.pdf")
            )
            with uploaded_source(file_url) as uploaded:
                pre_parsed.extend(_questions_from_file_bytes(file_name, uploaded))
        if bank_id and not pre_parsed:
            bank = store.get_question_bank(bank_id)
            if bank is None:
                raise HTTPException(404, "Past paper bank not found.")
            pre_parsed = list(bank.get("questions") or [])
    else:
        form = await request.form()
        document_id = str(form.get("document_id") or "").strip()
        questions_text = str(form.get("questions_text") or form.get("text") or "")
        f = form.get("file")
        if f is not None and hasattr(f, "read"):
            file_bytes = await _read_bounded_file(f)
            file_name = getattr(f, "filename", None) or "questions.txt"
            if (
                store._running_serverless()
                and len(file_bytes) > DIRECT_MULTIPART_SAFE_BYTES
            ):
                raise HTTPException(
                    413,
                    "On Vercel, questions files over ~4 MB must use Blob client upload "
                    "(the UI does this automatically when BLOB_READ_WRITE_TOKEN is set).",
                )

    if not document_id:
        raise HTTPException(400, "document_id is required. Upload your notes first.")

    doc = store.get_document(document_id)
    if doc is None:
        raise HTTPException(404, "Document not found. Upload your lecture notes again.")

    # Extract questions from optional file (multipart bytes or Blob download)
    from_file: list[str] = []
    if file_bytes:
        from_file = _questions_from_file_bytes(file_name or "questions.txt", file_bytes)

    from_paste = answer_from_notes.parse_questions(questions_text)
    # Prefer explicit list if provided (JSON), else merge paste + file
    if is_json and pre_parsed:
        questions = pre_parsed
    else:
        questions = from_paste + from_file
        # dedupe while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for q in questions:
            k = " ".join(q.lower().split())
            if k in seen:
                continue
            seen.add(k)
            uniq.append(q)
        questions = uniq

    if not questions:
        raise HTTPException(
            400,
            "No questions found. Paste questions in the text box and/or upload a "
            "questions file (PDF, Word, or text). Tip: number them 1. 2. 3. "
            "or put a blank line between each question.",
        )

    parsed_count = len(questions)
    truncated = parsed_count > answer_from_notes.MAX_QUESTIONS
    if truncated:
        questions = questions[: answer_from_notes.MAX_QUESTIONS]

    try:
        answers = answer_from_notes.answer_questions(
            questions=questions,
            chunks=list(doc["chunks"]),
            retriever=doc["retriever"],
            doc_title=doc["title"] or "Lecture material",
        )
    except generator.GenerationError as e:
        raise HTTPException(503, str(e))

    ok_count = sum(1 for a in answers if not a.error)
    source_text = doc.get("text") or "\n\n".join(doc["chunks"])
    answer_dicts = [a.model_dump() for a in answers]
    answer_dicts, g_summary = grounding.annotate_note_answers(
        answer_dicts, source_text
    )
    return {
        "mode": "my_questions",
        "document_id": document_id,
        "document_title": doc["title"],
        "parsed_count": parsed_count,
        "question_count": len(answers),
        "answered_ok": ok_count,
        "truncated": truncated,
        "max_questions": answer_from_notes.MAX_QUESTIONS,
        "answers": answer_dicts,
        "grounding": g_summary,
    }


# ---------------------------------------------------------------------------
# Study loop — safe quiz load, SRS, banks, share
# ---------------------------------------------------------------------------


@app.get("/api/quiz/{quiz_id}")
def get_quiz_client(quiz_id: str):
    """Client-safe quiz payload (no answers). Used for re-take and share links."""
    quiz = store.get_quiz(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz not found.")
    return {
        "quiz_id": quiz_id,
        "document_id": quiz.get("doc_id"),
        "use_rag": quiz.get("use_rag"),
        "topic": quiz.get("topic"),
        "difficulty": quiz.get("difficulty") if "difficulty" in quiz else None,
        "questions": [
            {"index": i, "question": q.question, "options": list(q.options)}
            for i, q in enumerate(quiz["questions"])
        ],
        "created_at": quiz.get("created_at"),
    }


@app.get("/api/study/due")
def study_due(document_id: str, limit: int = 10, card_type: str | None = None):
    if not document_id:
        raise HTTPException(400, "document_id is required.")
    if store.get_document(document_id) is None:
        raise HTTPException(404, "Document not found.")
    cards = store.list_due_srs_cards(
        document_id, limit=limit, card_type=card_type or None
    )
    # Client-safe for practice: include payload needed to show front; answers
    # only for flashcard flip (qa) — MCQ practice uses separate quiz path.
    safe = []
    for c in cards:
        p = dict(c.get("payload") or {})
        safe.append(
            {
                "id": c["id"],
                "card_type": c["card_type"],
                "due_at": c["due_at"],
                "front": p.get("question") or p.get("front") or "",
                "options": p.get("options"),
                # back only for non-mcq reveal; mcq graded server-side via review grade
                "back": p.get("back")
                or p.get("explanation")
                or (
                    (p.get("options") or [None])[p.get("correct_index", 0)]
                    if p.get("options")
                    else ""
                ),
                "reps": c["reps"],
            }
        )
    return {
        "document_id": document_id,
        "due_count": store.count_due_srs_cards(document_id),
        "cards": safe,
    }


@app.get("/api/study/wrong")
def study_wrong(document_id: str, limit: int = 20):
    if store.get_document(document_id) is None:
        raise HTTPException(404, "Document not found.")
    items = store.list_wrong_items(document_id, limit=limit)
    # Strip correct answers for list preview; full payload used when building quiz
    return {
        "document_id": document_id,
        "items": [
            {
                "question": it["payload"]["question"],
                "quiz_id": it["quiz_id"],
                "question_index": it["question_index"],
            }
            for it in items
        ],
        "count": len(items),
    }


class ReviewRequest(BaseModel):
    card_id: str
    grade: str = "good"  # again | hard | good | easy
    # Optional MCQ self-check: chosen option index (server can mark again/good)
    chosen_index: int | None = None


@app.post("/api/study/review")
def study_review(req: ReviewRequest):
    from . import study as study_mod

    card = store.get_srs_card(req.card_id)
    if card is None:
        raise HTTPException(404, "Card not found.")
    grade = (req.grade or "good").lower()
    payload = card.get("payload") or {}
    correct = None
    if req.chosen_index is not None and "correct_index" in payload:
        correct = int(req.chosen_index) == int(payload["correct_index"])
        grade = "good" if correct else "again"
    sched = study_mod.sm2_schedule(
        grade=grade,
        ease=card["ease"],
        interval_days=card["interval_days"],
        reps=card["reps"],
        lapses=card["lapses"],
    )
    store.update_srs_card_schedule(
        req.card_id,
        ease=sched["ease"],
        interval_days=sched["interval_days"],
        due_at=sched["due_at"],
        reps=sched["reps"],
        lapses=sched["lapses"],
    )
    return {
        "card_id": req.card_id,
        "grade": grade,
        "correct": correct,
        "schedule": sched,
        "explanation": payload.get("explanation") or "",
        "source_quote": payload.get("source_quote") or "",
        "correct_index": payload.get("correct_index"),
        "options": payload.get("options"),
    }


class QuizFromWrongRequest(BaseModel):
    document_id: str
    limit: int = 8


@app.post("/api/study/quiz-from-wrong")
def quiz_from_wrong(req: QuizFromWrongRequest):
    """Build a real stored quiz from weak/due SRS MCQ cards (no Claude)."""
    doc = store.get_document(req.document_id)
    if doc is None:
        raise HTTPException(404, "Document not found.")
    limit = max(1, min(req.limit, 15))
    cards = store.list_due_srs_cards(req.document_id, limit=limit, card_type="mcq")
    if len(cards) < 1:
        # Fall back to wrong items not yet in SRS
        wrong = store.list_wrong_items(req.document_id, limit=limit)
        questions = []
        groundings = []
        for it in wrong:
            p = it["payload"]
            questions.append(
                generator.QuizQuestion(
                    question=p["question"],
                    options=list(p["options"]),
                    correct_index=int(p["correct_index"]),
                    explanation=p.get("explanation") or "",
                    source_quote=p.get("source_quote") or "",
                )
            )
            groundings.append(
                grounding.QuestionGrounding(
                    grounded=False,
                    match_type="weak_bank",
                    source_quote=p.get("source_quote") or "",
                    options_unique=True,
                    expected_grounded=False,
                )
            )
    else:
        questions = []
        groundings = []
        for c in cards:
            p = c.get("payload") or {}
            opts = list(p.get("options") or [])
            if len(opts) != 4 or "correct_index" not in p:
                continue
            questions.append(
                generator.QuizQuestion(
                    question=p.get("question") or "",
                    options=opts,
                    correct_index=int(p["correct_index"]),
                    explanation=p.get("explanation") or "",
                    source_quote=p.get("source_quote") or "",
                )
            )
            groundings.append(
                grounding.QuestionGrounding(
                    grounded=False,
                    match_type="srs_card",
                    source_quote=p.get("source_quote") or "",
                    options_unique=True,
                    expected_grounded=False,
                )
            )
    if not questions:
        raise HTTPException(
            400,
            "No weak questions yet. Take a quiz first and miss some items.",
        )
    quiz_id = uuid.uuid4().hex[:12]
    empty_metrics = {
        "use_rag": False,
        "total_questions": len(questions),
        "grounded_count": 0,
        "ungrounded_count": len(questions),
        "quote_in_source_rate": 0.0,
        "options_unique_rate": 1.0,
        "match_type_counts": {},
        "empty_quote_rate": 0.0,
    }
    store.save_quiz(
        quiz_id=quiz_id,
        document_id=req.document_id,
        use_rag=False,
        topic="weak areas",
        questions=questions,
        groundings=groundings,
        pre_filter_metrics=empty_metrics,
        served_metrics=empty_metrics,
        context_chunks=[],
    )
    return {
        "quiz_id": quiz_id,
        "source": "weak_bank",
        "use_rag": False,
        "difficulty": "medium",
        "questions": [
            {"index": i, "question": q.question, "options": q.options}
            for i, q in enumerate(questions)
        ],
        "retrieval": {"strategy": "weak_bank", "num_batches": 0},
        "grounding": {
            "require_grounding": False,
            "filtered_out": 0,
            "best_effort": False,
            "warning": None,
        },
    }



# --- Exam writing ---


class ExamDraftRequest(BaseModel):
    answers: dict[str, str] = Field(default_factory=dict)
    attempt_id: str | None = None


@app.put("/api/exam/{exam_id}/draft")
def save_exam_draft(exam_id: str, req: ExamDraftRequest):
    if store.get_exam_paper(exam_id) is None:
        raise HTTPException(404, "Exam paper not found.")
    attempt_id = (req.attempt_id or "").strip() or uuid.uuid4().hex
    store.save_exam_attempt(
        attempt_id=attempt_id,
        exam_id=exam_id,
        answers=req.answers or {},
        status="draft",
    )
    return {"attempt_id": attempt_id, "status": "draft", "exam_id": exam_id}


@app.post("/api/exam/{exam_id}/submit-writing")
def submit_exam_writing(exam_id: str, req: ExamDraftRequest):
    if store.get_exam_paper(exam_id) is None:
        raise HTTPException(404, "Exam paper not found.")
    attempt_id = (req.attempt_id or "").strip() or uuid.uuid4().hex
    store.save_exam_attempt(
        attempt_id=attempt_id,
        exam_id=exam_id,
        answers=req.answers or {},
        status="submitted",
    )
    return {
        "attempt_id": attempt_id,
        "status": "submitted",
        "exam_id": exam_id,
        "message": "Answers saved. Generate marking guides to self-check side by side.",
    }


@app.get("/api/exam/{exam_id}/draft")
def get_exam_draft(exam_id: str):
    if store.get_exam_paper(exam_id) is None:
        raise HTTPException(404, "Exam paper not found.")
    draft = store.get_latest_exam_draft(exam_id)
    if draft is None:
        return {"exam_id": exam_id, "attempt_id": None, "answers": {}, "status": None}
    return draft


# --- Question banks ---


class BankCreate(BaseModel):
    title: str | None = None
    questions: list[str] | None = None
    questions_text: str | None = None
    document_id: str | None = None


@app.post("/api/banks")
def create_bank(req: BankCreate):
    qs: list[str] = []
    if req.questions:
        qs = [str(q).strip() for q in req.questions if str(q).strip()]
    elif req.questions_text:
        qs = answer_from_notes.parse_questions(req.questions_text)
    if not qs:
        raise HTTPException(400, "No questions to save.")
    bank_id = uuid.uuid4().hex[:12]
    title = (req.title or "").strip() or f"Past paper ({len(qs)} Qs)"
    store.save_question_bank(
        bank_id=bank_id,
        title=title,
        questions=qs,
        document_id=req.document_id,
    )
    return {
        "id": bank_id,
        "title": title,
        "question_count": len(qs),
    }


@app.get("/api/banks")
def list_banks():
    return store.list_question_banks()


@app.get("/api/banks/{bank_id}")
def get_bank(bank_id: str):
    row = store.get_question_bank(bank_id)
    if row is None:
        raise HTTPException(404, "Bank not found.")
    return row


@app.delete("/api/banks/{bank_id}")
def delete_bank(bank_id: str):
    if not store.delete_question_bank(bank_id):
        raise HTTPException(404, "Bank not found.")
    return {"ok": True, "id": bank_id}


class BankRename(BaseModel):
    title: str


@app.patch("/api/banks/{bank_id}")
def rename_bank(bank_id: str, body: BankRename):
    if not store.rename_question_bank(bank_id, body.title):
        raise HTTPException(404, "Bank not found.")
    return {"id": bank_id, "title": body.title.strip()}



# --- Share ---


class ShareCreate(BaseModel):
    resource_type: str  # quiz | exam
    resource_id: str
    password: str | None = None


@app.post("/api/share")
def create_share(req: ShareCreate):
    import hashlib

    rtype = (req.resource_type or "").strip().lower()
    rid = (req.resource_id or "").strip()
    if rtype not in ("quiz", "exam"):
        raise HTTPException(400, "resource_type must be quiz or exam.")
    if not store.resource_allowed(rtype, rid):
        raise security.WorkspaceAccessError()
    if rtype == "quiz" and store.get_quiz(rid) is None:
        raise HTTPException(404, "Quiz not found.")
    if rtype == "exam" and store.get_exam_paper(rid) is None:
        raise HTTPException(404, "Exam not found.")
    token = uuid.uuid4().hex
    pw_hash = None
    if req.password and req.password.strip():
        pw_hash = security.password_hash(req.password.strip())
    store.save_share_token(
        token=token,
        resource_type=rtype,
        resource_id=rid,
        password_hash=pw_hash,
    )
    return {
        "token": token,
        "resource_type": rtype,
        "resource_id": rid,
        "password_protected": bool(pw_hash),
        "path": f"?share={token}",
    }


class SharePassword(BaseModel):
    password: str = Field(default="", max_length=200)


@app.post("/api/share/{token}/redeem")
def resolve_share(token: str, body: SharePassword):
    day = datetime.now(timezone.utc).date().isoformat()
    if not store.consume_budget(f"share:global:{day}", 1000) or not store.consume_budget(f"share:{security.workspace.get()}:{day}", 20):
        raise HTTPException(429, "Share-link attempt limit reached.")
    result = store.redeem_share(token, body.password)
    if result is None:
        raise HTTPException(403, "Share link expired, unavailable, or password incorrect.")
    return result


@app.delete("/api/share/{token}")
def revoke_share(token: str):
    store.revoke_share(token)
    return {"ok": True}
