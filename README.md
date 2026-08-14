# StudyQuiz — Retrieval-Augmented Question Generation from Lecture Materials

Final year project — **Famule Oluwapamilerin Solomon (FTP/CSC/26/0133938)**, Department of Computer Science.
Supervisor: Dr. Aderibigbe.

StudyQuiz is an AI study assistant that generates practice assessments from a
student's own lecture PDFs. It uses **Retrieval-Augmented Generation (RAG)**: the
relevant parts of the uploaded document are retrieved and given to a large
language model (Claude), so every question is grounded in the actual course
material instead of hallucinated. A built-in **baseline mode** generates
questions *without* the retrieved context, which is used in the project's
evaluation chapter to compare grounded vs ungrounded question quality.

### Practice modes

| Mode | What you get | Scoring |
|------|----------------|---------|
| **MCQ Quiz** | Multiple-choice items with 4 options; weak-area re-quiz | Automatic |
| **Exam Quiz** | Nigerian-style theory paper with in-browser writing + marking guides | Self-check (not auto-marked) |
| **My questions** | Past papers from paste/upload; savable banks; answers from notes | Self-check |

**Share links:** after generate, copy `?quiz=` / `?exam=` (or create a `?share=` token). Classmates can take without re-uploading notes when storage is durable (Turso).

Exam Quiz mirrors written BSc papers (Discuss / Define / Explain with mark
allocations), so students can prepare for exams the way lecturers actually set
questions.

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
  evaluation rows. **Locally** it uses **SQLite** (`data/studyquiz.db`; override
  with `STUDYQUIZ_DB`). **On Vercel**, set **Turso** (`TURSO_DATABASE_URL` +
  `TURSO_AUTH_TOKEN`) so data survives cold starts; without Turso the app falls
  back to ephemeral `/tmp` SQLite.

## Running it

```bash
cd studyquiz
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # get one at https://platform.claude.com/
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 — upload lecture material (PDF, Word, PowerPoint,
or any text-based file, **up to 100 MB**), choose the number of questions,
difficulty, optional focus topic (with **auto-suggested chips** from chapter
and section headings in your notes), and take the quiz. MCQ generation
**retries ungrounded items only**, drops **near-duplicates**, and can use
**harder distractors**. The UI shows answer progress, weak-area re-quiz, and
**print / save PDF**.

### Upload size (important)

| Where | Practical max | How |
|--------|----------------|-----|
| **Local** (`uvicorn`) | **100 MB** | Direct multipart to `/api/documents` |
| **Vercel (live)** | **~100 MB** | Browser → **Vercel Blob** → API indexes by URL |
| **Vercel without Blob** | **~4.5 MB** | Serverless request body limit |

For large files on Vercel you **must** set `BLOB_READ_WRITE_TOKEN` (Storage → Blob
in the Vercel dashboard). The UI uses client upload for files over ~3.5 MB so
the file never goes through the Python function body.

> **Deploy note — durable storage (Turso):** On Vercel, local SQLite can only
> write to `/tmp` and **disappears on cold start**. For lasting documents and
> quizzes, create a free [Turso](https://turso.tech) database and set:
>
> | Variable | Example |
> |----------|---------|
> | `TURSO_DATABASE_URL` | `libsql://studyquiz-yourorg.turso.io` |
> | `TURSO_AUTH_TOKEN` | token from `turso db tokens create studyquiz` |
>
> Schema is created automatically on first request (same tables as local SQLite).
> Local `uvicorn` keeps using `data/studyquiz.db` unless Turso env vars are set.
>
> **Access PIN (recommended on public deploys):** set `STUDYQUIZ_ACCESS_PIN` so
> upload, generate, and evaluation routes require header `X-StudyQuiz-Pin`
> (the UI prompts once per session). Leave unset for open local demos.
>
> **Blob URL allowlist:** remote ingest only accepts Vercel Blob hosts
> (`*.public.blob.vercel-storage.com`, `*.blob.vercel-storage.com`). Extra hosts
> via `BLOB_ALLOWED_HOSTS` (comma-separated).
>
> Health check: `GET /api/health` reports `storage_backend`, `storage_durable`,
> `blob_configured`, `auth_required`, and `max_upload_mb`.

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
dropped when some still pass). If **every** quote fails, StudyQuiz **retries
once**, then serves a **best-effort** quiz with a warning instead of a hard
error. Untick the option to keep every model question and measure the raw
failure rate — useful for ablation tables.

## Scope / limitations (as stated in the proposal)

- PDF, Word, PowerPoint, plain text. **Scanned/image-only PDFs** use an optional
  **Claude OCR fallback** when embedded text is empty (requires `ANTHROPIC_API_KEY`;
  capped by `OCR_MAX_PAGES`, default 30). Handwriting quality varies.
- Multiple-choice questions only; essay grading is out of scope.
- Single-user store (no accounts yet); multi-user auth is future work.
  On Vercel, configure Turso for durable multi-instance storage (see deploy note).

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload and index a document |
| `GET` | `/api/documents` | List indexed documents |
| `PATCH` | `/api/documents/{id}` | Rename document (`{ "title" }`) |
| `PUT` | `/api/documents/{id}` | Replace file / re-index (same id; multipart or Blob URL) |
| `DELETE` | `/api/documents/{id}` | Delete document and linked quizzes / exams |
| `POST` | `/api/quiz` | Generate MCQ quiz (`document_id`, `num_questions`, `topic?`, `use_rag`, `require_grounding`, `difficulty`) |
| `POST` | `/api/quiz/{id}/submit` | Grade submitted MCQ answers |
| `POST` | `/api/exam` | Generate theory **Exam Quiz** paper (`num_questions` major questions, `course_code`, `course_title`, …) |
| `GET` | `/api/exam/{id}` | Fetch a saved exam paper |
| `POST` | `/api/exam/{id}/answers` | After you attempt the paper: marking points + model-answer outlines from your notes |
| `POST` | `/api/answer-from-notes` | Your own questions (paste and/or file; large files via Blob URL) → answers from notes |
| `GET` | `/api/quiz/{id}/evaluation` | Grounding metrics + quotes for one quiz |
| `GET` | `/api/evaluation/summary` | Aggregate RAG vs baseline rates |
| `GET` | `/api/evaluation/export` | CSV export of all evaluation rows |

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

Grounding tests do **not** call the Claude API.

## Project defense report

A full viva/defense write-up is in:

- [`docs/StudyQuiz_Project_Defense_Report.docx`](docs/StudyQuiz_Project_Defense_Report.docx)

Regenerate after doc changes:

```bash
npm install   # once — installs `docx`
node docs/generate_defense_report.js
```
