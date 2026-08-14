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
import io
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import (
    answer_from_notes,
    coverage,
    exam_generator,
    generator,
    grounding,
    pdf_processor,
    store,
)
from .retriever import BM25Retriever

app = FastAPI(title="StudyQuiz")

STATIC_DIR = Path(__file__).parent / "static"

# App-side ceiling (local multipart + blob-download path).
# On Vercel, multipart bodies are still capped by the platform (~4.5 MB);
# large files must use client → Vercel Blob → process-by-URL.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
# Guard extracted text so a huge PDF cannot explode SQLite / BM25 memory.
MAX_EXTRACTED_CHARS = int(os.getenv("MAX_EXTRACTED_CHARS", "5000000"))
# Direct multipart is only reliable under the Vercel body limit when live.
DIRECT_MULTIPART_SAFE_BYTES = 4 * 1024 * 1024


@app.on_event("startup")
def _startup() -> None:
    # Best-effort: Vercel cold starts may skip or re-run this. Schema is also
    # applied lazily on the first DB use so a startup failure cannot 500 the site.
    try:
        store.init_db()
    except OSError:
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
    except Exception as e:
        ok = False
        detail = f"{type(e).__name__}: {e}"
    return {
        "status": "ok" if ok else "degraded",
        "db_path": info["db_path"],
        "storage": info,
        "storage_backend": info["backend"],
        "storage_durable": info["durable"],
        "serverless": store._running_serverless(),
        "detail": detail,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "blob_configured": bool(os.getenv("BLOB_READ_WRITE_TOKEN")),
        "upload_hint": (
            "Use Vercel Blob client upload for files over ~4 MB on Vercel."
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
    """Download a remote file (e.g. Vercel Blob) with a size ceiling.

    Private Blob stores need the read-write token on the request.
    """
    max_bytes = MAX_UPLOAD_BYTES
    headers: dict[str, str] = {}
    blob_token = os.getenv("BLOB_READ_WRITE_TOKEN") or ""
    if blob_token and (
        "blob.vercel-storage.com" in url
        or "vercel-storage.com" in url
        or "public.blob.vercel-storage.com" in url
    ):
        headers["Authorization"] = f"Bearer {blob_token}"

    try:
        with httpx.Client(follow_redirects=True, timeout=180.0, headers=headers) as client:
            with client.stream("GET", url) as resp:
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            400,
            f"Failed to download file for indexing: {type(e).__name__}: {e}",
        ) from e


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
        data = _download_url_to_bytes(payload.url)
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
    data = await file.read()
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
    """Delete a document and related quizzes / exam papers / eval rows."""
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
        data = _download_url_to_bytes(payload.url)
        filename = payload.filename or _filename_from_url(payload.url)
        return _index_document_bytes(
            filename, data, replace_doc_id=doc_id
        )

    form = await request.form()
    file = form.get("file")
    if file is None or not hasattr(file, "read"):
        raise HTTPException(
            400,
            "Expected multipart field 'file', or JSON {url, filename} after "
            "Vercel Blob client upload.",
        )
    data = await file.read()
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
    )
    pre_groundings, pre_metrics = grounding.validate_quiz(
        quiz.questions,
        source_text=source_text,
        use_rag=use_rag,
    )
    return quiz, pre_groundings, pre_metrics


@app.post("/api/quiz")
def create_quiz(req: QuizRequest):
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

    try:
        quiz, pre_groundings, pre_metrics = _generate_mcq_attempt(
            doc=doc,
            num_questions=num_questions,
            plan=plan,
            context_batches=context_batches,
            topic=req.topic,
            use_rag=req.use_rag,
            difficulty=req.difficulty,
            source_text=source_text,
        )
    except generator.GenerationError as e:
        raise HTTPException(503, str(e))

    if not quiz.questions:
        raise HTTPException(
            503,
            "The model returned no questions. Try again or use a longer document.",
        )

    served_questions, served_groundings, filtered_out, best_effort = (
        grounding.resolve_grounded_serving(
            quiz.questions,
            pre_groundings,
            use_rag=req.use_rag,
            require_grounding=req.require_grounding,
        )
    )

    retried = False
    # If strict grounding wiped the whole set, auto-retry generation once.
    if best_effort and req.use_rag and req.require_grounding:
        try:
            quiz2, pre2, metrics2 = _generate_mcq_attempt(
                doc=doc,
                num_questions=num_questions,
                plan=plan,
                context_batches=context_batches,
                topic=req.topic,
                use_rag=req.use_rag,
                difficulty=req.difficulty,
                source_text=source_text,
            )
            if quiz2.questions:
                retried = True
                s2, g2, f2, be2 = grounding.resolve_grounded_serving(
                    quiz2.questions,
                    pre2,
                    use_rag=req.use_rag,
                    require_grounding=req.require_grounding,
                )
                # Prefer any attempt that has verified questions; else keep retry output.
                quiz, pre_groundings, pre_metrics = quiz2, pre2, metrics2
                served_questions, served_groundings, filtered_out, best_effort = (
                    s2,
                    g2,
                    f2,
                    be2,
                )
        except generator.GenerationError:
            # Keep first attempt (best-effort) rather than failing the student.
            pass

    post_metrics = grounding.summarise(served_groundings, use_rag=req.use_rag)

    warning = None
    if best_effort:
        total = len(served_questions)
        grounded_n = sum(1 for g in served_groundings if g.grounded)
        warning = (
            "Could not verify source quotes for these questions"
            f" ({grounded_n}/{total} matched the notes)"
            + (" after one automatic retry. " if retried else ". ")
            + "Serving best-effort questions — answers may be less tightly grounded. "
            "You can generate again or turn off “Only keep questions verified against the notes”."
        )

    quiz_id = uuid.uuid4().hex[:12]
    store.save_quiz(
        quiz_id=quiz_id,
        document_id=req.document_id,
        use_rag=req.use_rag,
        topic=req.topic,
        questions=served_questions,
        groundings=served_groundings,
        pre_filter_metrics=pre_metrics.to_dict(),
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
            questions=quiz.questions,
            groundings=pre_groundings,
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
            "pre_filter": pre_metrics.to_dict(),
            "served": post_metrics.to_dict(),
            "retried": retried,
            "best_effort": best_effort,
            "warning": warning,
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

    return {
        "score": score,
        "total": len(questions),
        "results": results,
        "grounding": quiz.get("served_metrics"),
    }


@app.get("/api/quiz/{quiz_id}/evaluation")
def quiz_evaluation(quiz_id: str):
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
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in eval_rows:
        writer.writerow(row)

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
    if not eval_rows:
        return {"quizzes": 0, "by_condition": {}}

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
    return {"by_condition": out}


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

    if is_json:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON body.")
        document_id = str(body.get("document_id") or "").strip()
        questions_text = str(body.get("questions_text") or body.get("text") or "")
        raw_list = body.get("questions")
        if isinstance(raw_list, list):
            pre_parsed = [str(q).strip() for q in raw_list if str(q).strip()]
        # Large questions files: browser → Vercel Blob → process by URL
        # (same path as lecture notes; avoids ~4.5 MB serverless body limit).
        file_url = str(
            body.get("questions_file_url") or body.get("file_url") or ""
        ).strip()
        if file_url:
            file_bytes = _download_url_to_bytes(file_url)
            file_name = (
                str(body.get("questions_filename") or body.get("filename") or "").strip()
                or _filename_from_url(file_url, "questions.pdf")
            )
    else:
        form = await request.form()
        document_id = str(form.get("document_id") or "").strip()
        questions_text = str(form.get("questions_text") or form.get("text") or "")
        f = form.get("file")
        if f is not None and hasattr(f, "read"):
            file_bytes = await f.read()
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
