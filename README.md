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
- **Coverage planning** — `coverage.py` scales how many chunks are retrieved
  with document length and quiz size (not a fixed top_k=6), diversifies topic
  hits so neighbouring windows are not over-sampled, and for long selections
  **batches generation by section** so questions span the whole lecture.
- **Generation** — Claude (`claude-haiku-4-5`) receives the retrieved chunks
  and returns questions as **structured output** validated against a Pydantic
  schema — guaranteed parseable JSON, each question carrying a supporting
  quote from the source material.
- **Grounding validation** — `grounding.py` checks each `source_quote` against
  the full document text (exact + punctuation-normalised match). Questions
  whose quote cannot be found are flagged; in default RAG mode they are
  dropped so students only see verified items. The **quote-in-source rate**
  is an automatic metric for the evaluation chapter (RAG vs baseline).
- **Grading** — answers are kept server-side; the browser never sees the
  correct option until the quiz is submitted.
- **Persistence** — `store.py` saves documents, quizzes, attempt scores, and
  evaluation rows in **SQLite** (`data/studyquiz.db` by default). Restart the
  server and previous uploads are still available. Override the path with
  `STUDYQUIZ_DB=/path/to/file.db`.

## Running it

```bash
cd studyquiz
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # get one at https://platform.claude.com/
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 — upload lecture material (PDF, Word, PowerPoint,
or any text-based file), choose the number of questions, difficulty, optional
focus topic and exam timer, and take the quiz. The UI shows answer progress,
warns on unanswered items, supports **review wrong only**, and **print / save
PDF**. Saved documents reappear in the dropdown after a restart.

> **Deploy note:** On **Vercel**, SQLite uses `/tmp/studyquiz.db` (the only
> writable path). Data lasts for the life of that serverless instance only —
> fine for demos, not for multi-user production. Locally the DB is
> `data/studyquiz.db`. A long-lived host or external DB is needed for durable
> multi-instance deploys. Health check: `GET /api/health`.

## Evaluation mode (for the project write-up)

Untick **"Ground questions in the document (RAG)"** in the UI to generate the
plain-LLM baseline for the same document. Collect quizzes from both conditions
and have raters score each question for *relevance*, *correctness*, and
*answerability from the source* — that comparison is the research contribution.

### Automatic metrics (no human raters required)

After each quiz is generated, StudyQuiz records:

| Metric | Meaning |
|---|---|
| **quote_in_source_rate** | Fraction of questions whose `source_quote` appears in the document |
| **options_unique_rate** | Fraction of questions with four distinct options |
| **match_type** | `exact` / `normalized` / `not_found` / `empty_quote` / `baseline_empty` |

Use these endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/quiz/{id}/evaluation` | Per-question grounding detail for one quiz |
| `GET` | `/api/evaluation/summary` | Aggregate RAG vs baseline quote-in-source rates |
| `GET` | `/api/evaluation/export` | Download all rows as **CSV** for tables/graphs |

Default RAG generation sets `require_grounding=true` (unverified questions are
dropped). Untick that option in the UI to keep every model question and measure
the raw failure rate — useful for ablation tables.

## Scope / limitations (as stated in the proposal)

- Text-based documents (PDF, Word, PowerPoint, plain text) — scanned/handwritten documents (OCR) are out of scope.
- Multiple-choice questions only; essay grading is out of scope.
- Single-user SQLite store (no accounts yet); multi-user auth is future work.
  Serverless hosts still need external storage for durable multi-instance use.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload and index a document |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/quiz` | Generate a quiz (`document_id`, `num_questions`, `topic?`, `use_rag`, `require_grounding`, `difficulty`) |
| `POST` | `/api/quiz/{id}/submit` | Grade submitted answers |
| `GET` | `/api/quiz/{id}/evaluation` | Grounding metrics + quotes for one quiz |
| `GET` | `/api/evaluation/summary` | Aggregate RAG vs baseline rates |
| `GET` | `/api/evaluation/export` | CSV export of all evaluation rows |

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

Grounding tests do **not** call the Claude API.
