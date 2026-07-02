"""StudyQuiz — Retrieval-Augmented Question Generation from lecture materials.

FastAPI application tying the pipeline together:
  upload → extract & chunk (pdf_processor) → index (retriever)
  → generate quiz (generator, Claude) → take quiz → server-side grading.

Run with:  uvicorn app.main:app --reload
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import generator, pdf_processor
from .retriever import BM25Retriever

app = FastAPI(title="StudyQuiz")

STATIC_DIR = Path(__file__).parent / "static"

# In-memory stores. Fine for a single-user demo; the report's "future work"
# section notes the swap to a database for multi-user deployment.
DOCUMENTS: dict[str, dict] = {}  # doc_id -> {title, chunks, retriever}
QUIZZES: dict[str, dict] = {}  # quiz_id -> {questions: [QuizQuestion], doc_id}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


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

    retriever = BM25Retriever()
    retriever.index(chunks)

    doc_id = uuid.uuid4().hex[:12]
    DOCUMENTS[doc_id] = {
        "title": file.filename,
        "chunks": chunks,
        "retriever": retriever,
    }
    return {
        "id": doc_id,
        "title": file.filename,
        "num_chunks": len(chunks),
        "num_words": len(text.split()),
    }


@app.get("/api/documents")
def list_documents():
    return [
        {"id": doc_id, "title": doc["title"], "num_chunks": len(doc["chunks"])}
        for doc_id, doc in DOCUMENTS.items()
    ]


class QuizRequest(BaseModel):
    document_id: str
    num_questions: int = 5
    topic: str | None = None
    use_rag: bool = True  # False = ungrounded baseline for the evaluation study


@app.post("/api/quiz")
def create_quiz(req: QuizRequest):
    doc = DOCUMENTS.get(req.document_id)
    if doc is None:
        raise HTTPException(404, "Document not found. Upload it again.")
    num_questions = max(1, min(req.num_questions, 15))

    # Retrieval: topic query if given, otherwise sample evenly across the
    # document so the quiz covers all of the material.
    retriever: BM25Retriever = doc["retriever"]
    if req.topic:
        hits = retriever.search(req.topic, top_k=6)
        if not hits:
            hits = retriever.spread_sample(top_k=6)
    else:
        hits = retriever.spread_sample(top_k=6)
    context_chunks = [doc["chunks"][i] for i, _score in hits]

    try:
        quiz = generator.generate_quiz(
            num_questions=num_questions,
            doc_title=doc["title"],
            context_chunks=context_chunks,
            topic=req.topic,
            use_rag=req.use_rag,
        )
    except generator.GenerationError as e:
        raise HTTPException(503, str(e))

    quiz_id = uuid.uuid4().hex[:12]
    QUIZZES[quiz_id] = {"questions": quiz.questions, "doc_id": req.document_id}

    # Answers stay server-side; the client only sees questions and options.
    return {
        "quiz_id": quiz_id,
        "questions": [
            {"index": i, "question": q.question, "options": q.options}
            for i, q in enumerate(quiz.questions)
        ],
    }


class SubmitRequest(BaseModel):
    answers: list[int]  # chosen option index per question, -1 = unanswered


@app.post("/api/quiz/{quiz_id}/submit")
def submit_quiz(quiz_id: str, req: SubmitRequest):
    quiz = QUIZZES.get(quiz_id)
    if quiz is None:
        raise HTTPException(404, "Quiz not found.")
    questions = quiz["questions"]
    if len(req.answers) != len(questions):
        raise HTTPException(400, "Answer count does not match question count.")

    results = []
    score = 0
    for q, chosen in zip(questions, req.answers):
        correct = chosen == q.correct_index
        score += correct
        results.append(
            {
                "correct": correct,
                "chosen_index": chosen,
                "correct_index": q.correct_index,
                "explanation": q.explanation,
                "source_quote": q.source_quote,
            }
        )
    return {"score": score, "total": len(questions), "results": results}
