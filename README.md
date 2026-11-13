# StudyQuiz — Retrieval-Augmented Question Generation from Lecture Materials

Final year project — **Famule Oluwapamilerin Solomon**, Department of Computer Science.
Supervisor: Dr. Aderibigbe.

StudyQuiz creates practice quizzes, theory papers and answers from lecture notes.
It retrieves relevant passages with BM25 and supplies them to Claude. RAG can
help relevance; it does **not** guarantee correct questions or answers.

## Practice modes

| Mode | Output | Assessment |
|---|---|---|
| MCQ quiz | Four-option questions; explanations after submission | Automatic comparison to the generated answer key |
| Exam quiz | Theory paper, private writing draft and optional marking guides | Self-check, not automatic marking |
| My questions | Answers to pasted/uploaded past questions; saved question banks | Self-check |

This is a study tool, not a secure examination platform. Generated answer keys
can be wrong. “Quote found” means quoted text occurs in the notes, not that the
quote proves the answer.

## Local development

Use **Python 3.12** and **Node 22** (Node is needed for the deployed upload handler
and its tests). Local files use a persistent SQLite database at
`data/studyquiz.db`; `STUDYQUIZ_DB` overrides the path.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
# Set ANTHROPIC_API_KEY in your environment; never commit it.
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`. Local browser uploads use multipart, including large
files, without needing the Vercel Node route. Documents may be PDF, DOCX, PPTX or
readable text. Maximum upload: 100 MB by default; PDFs are capped at 200 pages,
extracted text at 5 million characters, and Office archive expansion is bounded.
Scanned PDF OCR sends scanned pages to Anthropic and consumes the AI allowance.

Local demos may run without a PIN. Set `STUDYQUIZ_ACCESS_PIN` to require one.
`.env.example` lists settings, but the app does not automatically load `.env`.
Load environment variables yourself; never share secret values in an issue.

## Public Vercel deployments

Public deployments fail closed unless access protection and durable storage are
configured. They no longer silently save work to disposable `/tmp` storage.

| Setting | Purpose |
|---|---|
| `STUDYQUIZ_ACCESS_PIN` | Required access secret of at least 16 characters; enter it in the inline Unlock workspace form |
| `ANTHROPIC_API_KEY` | Claude generation/OCR |
| `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN` | Durable shared database across workers |
| `STUDYQUIZ_PRIVATE_BLOB_READ_WRITE_TOKEN` | Isolated token for this application's **private** Blob store; legacy `BLOB_READ_WRITE_TOKEN` is a fallback |
| `BLOB_ACCESS=private` | Public upload mode is refused |
| `AI_DAILY_GLOBAL_CALLS=100` | Deployment-wide maximum AI calls per UTC day |
| `AI_DAILY_WORKSPACE_CALLS=30` | Maximum per browser workspace per UTC day |
| `MAX_UPLOAD_MB=100` | Upload limit, capped at 100 MB |
| `MAX_DOCUMENT_PAGES=200` | PDF page limit before extraction/OCR |
| `MAX_EXTRACTED_CHARS=5000000` | Extracted text ceiling |

Large live uploads go directly to private Blob storage. The separate Node route
checks the PIN, then obtains a durable, workspace-bound upload permit before
issuing a short-lived upload token. It calls the deployment's own
`https://VERCEL_URL/api/blob/permit`; deployment protection must allow this
server-to-server call. It never trusts a request's Host header for that URL.
There is no public/private fallback. Both serverless runtimes must receive the
same PIN and matching environment settings.

Uploaded URLs must belong to this store and an upload permit owned by the current
workspace. Redirects to other stores, arbitrary download hosts, embedded URL
credentials, and nonstandard ports are rejected. The store token is never sent
to another store or a custom allowed host.

AI budgets count individual calls, including batches, OCR and retries, with
atomic durable counters. SDK automatic retries are disabled. They are call
limits, **not a dollar-denominated billing cap**; also configure provider limits.
A deployment gets at most 100 upload permits/day and a workspace 20/day.
Failed/reserved attempts conservatively consume allowances.

## Workspaces, sharing and privacy

Each browser receives an opaque HttpOnly, SameSite=Lax workspace cookie. Saved
notes, quizzes, banks, drafts and evaluation data are scoped to its hashed ID.
The PIN gates deployment access; it is **not** a personal login. Clearing cookies,
using another browser, or changing domains does not recover the original
workspace. A full account/recovery system is outside this change.

Use the UI's **Copy link** buttons to create a scoped 7-day practice link.
A bare `?quiz=` or `?exam=` ID does not grant another workspace access.
Optional share passwords can be set through `POST /api/share`; they are salted
PBKDF2 hashes and are submitted in a POST body, never a URL. Redeeming a share
link creates a browser-bound grant to that single resource. Recipients have
separate drafts and cannot change someone else's attempt or access the owner's
notes, exports, or paid generation. `DELETE /api/share/{token}` revokes the link
and its grants. Legacy unsalted/password links must be recreated.

Relevant note text, and scanned pages for OCR, are sent to Anthropic. The UI
states this before upload. The app saves extracted text for study. Original Blob
files are deleted after processing, including extraction failures. Failed
cleanup stays in a durable queue, is retried on the next upload after one hour,
and can be retried with `POST /api/uploads/cleanup`. Abandoned uploads are also
covered by that queue. Operators should schedule regular cleanup if users may
never return. No unattended cleanup schedule is created by this repository.

If original-file deletion still cannot be confirmed, deleting its document
returns a retryable error and retains the saved record so cleanup is not lost.
Successful document deletion removes related evaluation runs/reviews and saved
work. Previously uploaded public files are **not automatically discovered** by
this upgrade: audit and remove them separately in the old Blob store.

### Existing data upgrade

Back up the database first. Existing records have no trustworthy owner, so they
remain unclaimed rather than becoming visible to the next visitor. New tables
are created additively; existing source rows are not erased by startup.

1. Deploy with the required PIN/private storage/Turso settings on a preview.
2. Open the app in the browser that should own the legacy data.
3. Obtain its workspace ID from authenticated `GET /api/workspace`.
4. Run the operator-only migration against the intended database, first as a dry
   run, then with `--apply` after reviewing the counts:

```sh
python scripts/claim_legacy_workspace.py --workspace-id YOUR_64_CHARACTER_ID
python scripts/claim_legacy_workspace.py --workspace-id YOUR_64_CHARACTER_ID --apply
```

It assigns only unclaimed records and never reassigns existing ownership. This
operation is intended for the old single-user dataset. Do not assign a mixed
multi-student dataset to one person. Recreate old share links after migration.
Do not run this against production without a verified backup and intended owner.

## Evaluation for the final-year report

The baseline receives title/topic without retrieved text; its prompt does not
request quotes. The validator now checks any supplied quote identically in both
conditions; it no longer forces baseline scores to zero. Nevertheless, comparing
quote rates is **not** an accuracy experiment because the prompts have different
access to evidence. The summary labels this limitation explicitly.

Strict quote mode returns an insufficient-evidence error when no question passes
quote and option checks. It never silently serves unverified questions. Users
may explicitly disable the filter. Duplicate options are rejected by the strict
filter. All MCQ attempts/retries, including all-failed runs, are retained
separately from served output, with elapsed time and status. Original pre-filter
rows are preserved before duplicate removal/replacement; served rows are a
separate phase. Old records are not retroactively re-evaluated.

For correctness, use the same independent human rubric in both conditions:

- Is the marked answer correct?
- Does the evidence actually support it?
- Is the question clear and unambiguous?

`PUT /api/quiz/{id}/reviews/{index}` records those three Boolean judgements plus
optional notes. Unreviewed accuracy remains null, not zero or “verified.”
`GET /api/evaluation/summary` reports human-review counts and rates separately
from quote provenance. The CSV includes review fields on served questions and
neutralizes spreadsheet formula prefixes. [Evaluation protocol](docs/evaluation-protocol.md)
describes a fair experiment. Existing defense DOCX files are historical drafts;
update the generator/report to match this implementation and actual results
before submitting them. Do not treat old claims or metrics as validated results.

## Tests

```sh
python -m pytest tests -q
npm ci --ignore-scripts
node --test tests/authentication.test.cjs tests/blob-upload.test.cjs
```

Tests use isolated local databases, fake model responses and mocked upload
services. They do not make real uploads or paid AI calls. Use Python 3.12
and Node 22 for automated checks. GitHub workflow setup is separate because
the connected OAuth app cannot create workflow files. Live private-store and
Turso configuration still require a preview smoke test using operator-controlled
test data.
# Student accounts, dashboard and progress

Student sign-in is optional until configured and verified. The existing PIN
deployment continues unchanged when the new variables are absent.

Configure a **separate StudyQuiz Supabase Auth project** (not another business's
project), with email authentication enabled:

- `STUDYQUIZ_AUTH_URL`: its HTTPS `https://<project>.supabase.co` URL.
- `STUDYQUIZ_AUTH_PUBLIC_KEY`: its publishable/anon key, **not** a service-role key.
- Vercel's native integration variables `NEXT_PUBLIC_SUPABASE_URL` and
  `NEXT_PUBLIC_SUPABASE_ANON_KEY` are also accepted, without copying credentials.
- `STUDYQUIZ_AUTH_MODE=accounts`: require verified student sessions instead of
  accepting a shared workspace code. Leave unset while testing in Preview.

Configure the email template to display `{{ .Token }}` rather than only a magic
link. Keep email verification enabled, set a short OTP expiry (10 minutes), and
configure an authorized SMTP sender before admitting public students. Supabase's
default mail sender has restrictions; validate delivery to an external student
address and confirm the sending provider's allowance before release. Do not
silently enable paid email or unrelated projects' mail credentials.

The application verifies email codes with Supabase, then fetches the verified
user before creating a StudyQuiz session. Provider access/refresh tokens are not
returned to the browser or retained. StudyQuiz sessions expire after 12 hours,
use HttpOnly/SameSite cookies (Secure on public deployments), and store only
token hashes in Turso. Sign out on one or all devices revokes them immediately.
The email inbox is the account recovery mechanism; no shared access code is
required for ordinary student use after account mode is enabled.

The dashboard lists saved notes, weighted MCQ quiz scores, recent attempts,
topic totals and suggested revision below 70%. Repeated attempts count
separately; exam self-assessments are excluded and scores are not claims of
mastery. Summary totals include all owned attempts, even when the recent list
is limited to 20. Browser-only work is never silently assigned to a new account:
the signed-in student can explicitly import the current browser's old workspace
using its previous deployment code. No supplied workspace ID is accepted.

Release checks: verify email delivery, inbox code expiry, two browsers for the
same account, isolation between students, logout and expiry, authenticated large
private upload, quiz scoring, dashboard reload, and explicit browser import in
Preview. Enable accounts in Production only after those checks pass. Keep the
old deployment code privately available for optional legacy import.
