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
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import coverage, generator, grounding, pdf_processor, store
from .retriever import BM25Retriever

app = FastAPI(title="StudyQuiz")

STATIC_DIR = Path(__file__).parent / "static"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


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
    try:
        store.init_db()
        db = str(store.db_path())
        ok = True
        detail = "ok"
    except Exception as e:
        ok = False
        db = str(store.db_path())
        detail = f"{type(e).__name__}: {e}"
    return {
        "status": "ok" if ok else "degraded",
        "db_path": db,
        "serverless": store._running_serverless(),
        "detail": detail,
    }


@app.post("/api/documents")
async def upload_document(file: UploadFile):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 20 MB).")
    try:
        text = pdf_processor.extract_text(file.filename or "upload.pdf", data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(
            400,
            "Could not read this file. Make sure it is a valid, non-corrupted "
            "document (PDF, Word, PowerPoint, or a text-based file).",
        )

    chunks = pdf_processor.chunk_text(text)
    if len(chunks) == 0:
        raise HTTPException(
            400,
            "No text could be extracted. Scanned/image-only documents are not "
            "supported (OCR is outside the project scope).",
        )

    doc_id = uuid.uuid4().hex[:12]
    store.save_document(
        doc_id=doc_id,
        title=file.filename or "untitled",
        text=text,
        chunks=chunks,
    )
    return {
        "id": doc_id,
        "title": file.filename,
        "num_chunks": len(chunks),
        "num_words": len(text.split()),
    }


@app.get("/api/documents")
def list_documents():
    return store.list_documents()


class QuizRequest(BaseModel):
    document_id: str
    num_questions: int = 5
    topic: str | None = None
    use_rag: bool = True  # False = ungrounded baseline for the evaluation study
    # When True (default) and use_rag=True, drop questions whose source_quote
    # is not found in the document so students only see verified items.
    # Evaluation metrics still record the full pre-filter generation.
    require_grounding: bool = True


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
        quiz = generator.generate_quiz_from_batches(
            num_questions=num_questions,
            doc_title=doc["title"],
            context_batches=context_batches,
            questions_per_batch=plan.questions_per_batch,
            topic=req.topic,
            use_rag=req.use_rag,
        )
    except generator.GenerationError as e:
        raise HTTPException(503, str(e))

    # --- Grounding validation (automatic evaluation metric) ---
    pre_groundings, pre_metrics = grounding.validate_quiz(
        quiz.questions,
        source_text=source_text,
        use_rag=req.use_rag,
    )

    served_questions = list(quiz.questions)
    served_groundings = list(pre_groundings)
    filtered_out = 0

    if req.use_rag and req.require_grounding:
        served_questions, served_groundings = grounding.filter_grounded(
            quiz.questions, pre_groundings
        )
        filtered_out = len(quiz.questions) - len(served_questions)
        if not served_questions:
            raise HTTPException(
                503,
                "No questions could be verified against the document "
                f"(0/{len(quiz.questions)} quotes found in the source). "
                "Try generating again, or turn off 'Require grounded quotes'.",
            )

    post_metrics = grounding.summarise(served_groundings, use_rag=req.use_rag)

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
