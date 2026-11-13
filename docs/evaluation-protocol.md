# Evaluation protocol

The application checks quote provenance automatically. This is distinct from
correctness, relevance and learning outcomes. No measured accuracy is claimed
until an independent evaluation has actually been conducted.

1. Freeze a held-out set of lecture documents/topics before tuning retrieval or
   prompts. Record document hashes, code commit, model name, model settings,
   date, chunking settings and retrieval settings.
2. Generate equal requested counts from RAG and title/topic-only baseline, using
   the same topics and difficulty. Run multiple independent repetitions. Save
   failures, empty outputs and retries instead of silently replacing them.
3. Keep original model output, filter decisions and served output separate.
   Report requested/generated/served counts and failed-run counts. A filtered
   or missing question is not evidence of accuracy. Report coverage as well.
4. Randomize and blind question order to independent subject-knowledgeable
   reviewers. Give reviewers the source material for both conditions. Evaluate
   marked-answer correctness, supporting evidence, clarity, relevance and
   distractor plausibility. Define how disagreements are resolved and report
   reviewer agreement where more than one reviewer participates.
5. Use the review API for the supported correctness/evidence/clarity rubric;
   store additional judgement dimensions in a versioned external study sheet.
   Never record a machine quote match as a human correctness judgement.
6. Report denominators: reviewed questions, unreviewed questions, all failures,
   original questions and served questions. Include elapsed time and number of
   calls/retries; use provider usage records if reporting token costs.
7. Report uncertainty and limitations. The baseline has no source context, so
   quote-in-source rate measures access/provenance, not comparative accuracy.
   Prompt/retrieval/model changes require a new experiment version.

Legacy records may have forced-zero baseline scores or missing all-failed runs.
Do not mix them into the new experiment without explicitly identifying those
limitations. Raw evaluation records can contain note excerpts and must remain
private to their workspace/research operator.
