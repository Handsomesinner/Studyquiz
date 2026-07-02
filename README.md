# StudyQuiz — Retrieval-Augmented Question Generation from Lecture Materials

Final year project — **Famule Oluwapamilerin Solomon (FTP/CSC/26/0133938)**, Department of Computer Science.
Supervisor: Dr. Aderibigbe.

StudyQuiz is an AI study assistant that generates multiple-choice quizzes from a
student's own lecture PDFs. It uses **Retrieval-Augmented Generation (RAG)**: the
relevant parts of the uploaded document are retrieved and given to a large
language model (Claude), so every question is grounded in the actual course
material instead of hallucinated. A built-in **baseline mode** generates
questions *without* the retrieved context, which is used in the project's
evaluation chapter to compare grounded vs ungrounded question quality.

## Architecture (maps to Chapter 3 of the report)

```
 ┌────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌─────────────────┐
 │  Upload    │──▶│ (i) Extraction & │──▶│ (ii) Retrieval │──▶│ (iii) Question  │
 │  PDF/TXT   │   │     chunking     │   │  (BM25 index)  │   │  generation     │
 └────────────┘   │ pdf_processor.py │   │  retriever.py  │   │  (Claude API)   │
                  └──────────────────┘   └────────────────┘   │  generator.py   │
                                                              └────────┬────────┘
                                          ┌────────────────┐           │
                                          │ (iv) Web quiz  │◀──────────┘
                                          │ UI + grading   │
                                          │ main.py+static │
                                          └────────────────┘
```

- **Extraction & chunking** — `pypdf` extracts text; a sliding word-window
  (200 words, 40-word overlap) splits it into retrievable chunks.
- **Retrieval** — the BM25 ranking function, implemented from scratch in
  `app/retriever.py` so the mathematics can be presented in the report. The
  class exposes the same interface an embedding + vector-database retriever
  (e.g. ChromaDB) would, so that upgrade is a drop-in replacement.
- **Generation** — Claude (`claude-haiku-4-5`) receives the retrieved chunks
  and returns questions as **structured output** validated against a Pydantic
  schema — guaranteed parseable JSON, each question carrying a supporting
  quote from the source material.
- **Grading** — answers are kept server-side; the browser never sees the
  correct option until the quiz is submitted.

## Running it

```bash
cd study-assistant
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # get one at https://platform.claude.com/
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 — upload a lecture PDF (or .txt/.md), choose the
number of questions and an optional focus topic, and take the quiz.

## Evaluation mode (for the project write-up)

Untick **"Ground questions in the document (RAG)"** in the UI to generate the
plain-LLM baseline for the same document. Collect quizzes from both conditions
and have raters score each question for *relevance*, *correctness*, and
*answerability from the source* — that comparison is the research contribution.

## Scope / limitations (as stated in the proposal)

- English, text-based PDFs only — scanned/handwritten documents (OCR) are out of scope.
- Multiple-choice questions only; essay grading is out of scope.
- In-memory storage (single-user demo); a database is listed as future work.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload and index a document |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/quiz` | Generate a quiz (`document_id`, `num_questions`, `topic?`, `use_rag`) |
| `POST` | `/api/quiz/{id}/submit` | Grade submitted answers |
