# Security and reliability fixes — 13 November 2026

Implemented locally on `fix/security-and-quiz-reliability`, based on feature
commit `fdde56d1431c0eaff78f84d4b035c804004755ac`. This deliberately includes
the newer persistence, OCR, exams and question-bank work instead of restoring
the old main branch's in-memory implementation.

| Audit item | Change |
|---|---|
| 1. Upload PIN bypass | Node token route checks the PIN; fails closed; private-only; short-lived grants; scoped upload permits; durable upload quotas; signed callback validation through the actual SDK |
| 2. Draft exposure/overwrite | Opaque HttpOnly browser workspace cookie; ownership checks at persistence boundaries; drafts queried per workspace; supplied IDs cannot claim other/legacy attempts; scoped, expiring and revocable shared-resource grants |
| 3. Public/retained source files | Public upload fallback removed; original Blob deleted after processing or failure; durable cleanup queue; deletion retries; source tracked until cleanup confirmed; document deletion also removes evaluation runs/reviews |
| 4. Anonymous documents, answers and costs | Protected APIs require access; exports/documents/banks/quizzes isolated by workspace; grants expose only the intended practice resource; durable per-call AI allowances across retries and worker threads |
| 5. Unsafe downloads | Preserve feature-branch URL protection; require a permit for this workspace and exact store; standard HTTPS port; token never forwarded to custom/other stores; cross-store redirects blocked |
| 6. Misleading quiz evaluation | Same quote validator for both modes; no forced-zero baseline; provenance labelled separately from correctness; original and served rows separate; failures/retries retained and visible; independent human rubric/API; CSV formula protection |
| 7. Reliability | Strict mode returns insufficient evidence rather than unverified fallback; duplicate options excluded; empty-term BM25 and one-chunk sampling fixed; durable serverless storage required; bounded parsing; local multipart browser path restored |

## Validation

- Python 3.12: **90 tests passed**, including 18 new boundary/regression tests.
- Node: **9 tests passed**, including actual installed-SDK token signing and
  unsigned-callback rejection with synthetic credentials and no network.
- `npm audit`: **0 known vulnerabilities** after the Blob SDK update and patched
  transitive `undici`/`zod` overrides. This is not a guarantee of no vulnerabilities.
- Python dependency consistency, JavaScript syntax and patch whitespace checks passed.
- No paid AI calls, real storage writes, or production database changes were made.
- Two existing FastAPI startup-event deprecation warnings remain; they do not
  fail the tests.

## Required release preparation

1. Back up the existing database and confirm which branch is currently deployed.
2. Set a deployment access secret of at least 16 characters. Public API routes
   deliberately return a configuration error if it is absent or too short.
3. Configure durable Turso storage and a **private** Blob store. A public Blob
   store cannot simply become private by changing the environment variable.
4. Assign legacy data to its intended browser workspace using the dry-run-first
   migration script. Nothing is automatically assigned to the next visitor.
5. Recreate old share links and audit/delete historical public Blob files. Old
   files are not discoverable from the database if their URLs were never stored.
6. Smoke-test a preview: small multipart upload, large private upload, source-file
   cleanup, quiz generation, owner/guest drafts, share expiry/revocation, and
   persistence across a redeployment. The Node upload function must be able to
   call the deployment's Python permit endpoint through deployment protection.

These configuration and live-service checks were not performed because no
Studyquiz production service access was supplied. Collaborator access was
subsequently granted for pushing the fixes branch. The OAuth connection cannot
create GitHub workflow files, so the optional automated-test workflow is supplied
separately for manual upload. Nothing was deployed.

## Explicit limits

Browser workspaces are separate capabilities, not full named user accounts.
Clearing the cookie or changing device/domain removes automatic access to saved
work; an operator needs to handle recovery. The shared deployment PIN is not an
individual user identity. Source text/pages sent to Anthropic remain subject to
that provider's retention settings. Cleanup retries are opportunistic or manual;
regular unattended maintenance still needs an operator schedule. Quote matching
cannot prove semantic correctness. Human review must actually be conducted before
making accuracy claims. This remains a self-study application, not a secure
high-stakes examination system.

See the root README for configuration and migration commands and
`docs/evaluation-protocol.md` for the experiment protocol.
