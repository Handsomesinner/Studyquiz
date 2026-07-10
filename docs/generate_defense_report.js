/**
 * StudyQuiz — comprehensive project defense report (DOCX).
 * Run: node docs/generate_defense_report.js
 */
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, LevelFormat, PageBreak,
} = require("docx");
const fs = require("fs");
const path = require("path");

// A4 content width ≈ 9026 DXA with ~1" margins (11906 - 2*1440)
const PAGE_W = 11906;
const PAGE_H = 16838;
const MARGIN = 1134; // 0.79"
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9638
const thin = { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" };
const borders = { top: thin, bottom: thin, left: thin, right: thin };
const headerBorder = { style: BorderStyle.SINGLE, size: 4, color: "0D6E5F" };
const headerBorders = { top: headerBorder, bottom: headerBorder, left: headerBorder, right: headerBorder };

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 160, before: opts.before ?? 0, line: opts.line ?? 276 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    ...opts.para,
    children: [
      new TextRun({
        text,
        font: "Times New Roman",
        size: opts.size || 24, // 12pt
        bold: !!opts.bold,
        italics: !!opts.italics,
      }),
    ],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, font: "Times New Roman", size: 28, bold: true })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, font: "Times New Roman", size: 26, bold: true })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, font: "Times New Roman", size: 24, bold: true })],
  });
}

function center(text, opts = {}) {
  return p(text, { ...opts, align: AlignmentType.CENTER });
}

function bullet(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80, line: 276 },
    children: [new TextRun({ text, font: "Times New Roman", size: 24 })],
  });
}

function num(text, ref = "numbers") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80, line: 276 },
    children: [new TextRun({ text, font: "Times New Roman", size: 24 })],
  });
}

function cell(text, width, opts = {}) {
  return new TableCell({
    borders: opts.header ? headerBorders : borders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.header
      ? { fill: "0D6E5F", type: ShadingType.CLEAR }
      : opts.alt
        ? { fill: "F6F4EE", type: ShadingType.CLEAR }
        : undefined,
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    children: [
      new Paragraph({
        children: [
          new TextRun({
            text,
            font: "Times New Roman",
            size: opts.size || 20,
            bold: !!opts.bold || !!opts.header,
            color: opts.header ? "FFFFFF" : "000000",
          }),
        ],
      }),
    ],
  });
}

function table(headers, rows, colWidths) {
  const w = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: w, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        children: headers.map((h, i) => cell(h, colWidths[i], { header: true, bold: true })),
      }),
      ...rows.map(
        (r, ri) =>
          new TableRow({
            children: r.map((c, i) => cell(String(c), colWidths[i], { alt: ri % 2 === 1 })),
          })
      ),
    ],
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

const children = [
  // ========== TITLE PAGE ==========
  new Paragraph({ spacing: { after: 200 }, children: [] }),
  center("FEDERAL UNIVERSITY OYE-EKITI", { bold: true, size: 28, after: 80 }),
  center("(FUOYE)", { bold: true, size: 24, after: 200 }),
  center("FACULTY OF SCIENCE", { bold: true, size: 24, after: 60 }),
  center("DEPARTMENT OF COMPUTER SCIENCE", { bold: true, size: 24, after: 360 }),
  center("PROJECT DEFENSE REPORT", { bold: true, size: 26, after: 200 }),
  center("StudyQuiz:", { bold: true, size: 32, after: 80 }),
  center("A Retrieval-Augmented Generation System for", { bold: true, size: 26, after: 40 }),
  center("Lecture-Grounded Multiple-Choice and Theory", { bold: true, size: 26, after: 40 }),
  center("Examination Practice", { bold: true, size: 26, after: 400 }),
  center("BY", { size: 22, after: 200 }),
  center("FAMULE OLUWAPAMILERIN SOLOMON", { bold: true, size: 26, after: 80 }),
  center("Matriculation Number: FTP/CSC/26/0133938", { size: 22, after: 300 }),
  center("Supervisor: Dr. Aderibigbe", { size: 22, after: 200 }),
  center("A Final Year Project Report Submitted in Partial Fulfilment", { size: 20, after: 40 }),
  center("of the Requirements for the Award of the Degree of", { size: 20, after: 40 }),
  center("Bachelor of Science (B.Sc.) in Computer Science", { size: 20, after: 300 }),
  center("July 2026", { size: 22, after: 100 }),

  pageBreak(),

  // ========== ABSTRACT ==========
  h1("ABSTRACT"),
  p(
    "Students in Nigerian universities often prepare for examinations using lecturer-provided notes, yet automated quiz tools frequently generate questions from general model knowledge rather than from the student’s own materials. This mismatch produces questions that are not answerable from the course pack and undermines trust in AI study tools. StudyQuiz addresses this problem with a retrieval-augmented generation (RAG) pipeline that (i) extracts and chunks uploaded lecture documents, (ii) ranks relevant passages with the BM25 ranking function implemented from first principles, (iii) generates questions with a large language model (Claude Haiku) constrained to structured JSON, and (iv) validates that claimed source quotes actually appear in the document."
  ),
  p(
    "The system offers two practice modes aligned with real assessment practice: multiple-choice quizzes with server-side grading, and Exam Quiz theory papers in the style of Nigerian BSc written examinations (QUESTION ONE/TWO, lettered parts, roman sub-parts, and mark allocations), with optional post-attempt marking guides grounded in the notes. Automatic evaluation metrics—especially quote-in-source rate—support comparison of RAG versus an ungrounded baseline for the research contribution of the project."
  ),
  p(
    "StudyQuiz is implemented as a FastAPI web application with a single-page interface, SQLite persistence, automated unit tests, and optional deployment on Vercel. This report documents the problem, objectives, design, implementation, evaluation approach, limitations, and defense talking points in detail."
  ),
  p("Keywords: Retrieval-Augmented Generation (RAG); BM25; educational assessment; question generation; grounding validation; FastAPI; Nigerian university examinations.", {
    italics: true,
  }),

  pageBreak(),

  // ========== TOC note ==========
  h1("TABLE OF CONTENTS"),
  p("1. Introduction", { align: AlignmentType.LEFT }),
  p("2. Problem Statement", { align: AlignmentType.LEFT }),
  p("3. Aim and Objectives", { align: AlignmentType.LEFT }),
  p("4. Scope and Limitations", { align: AlignmentType.LEFT }),
  p("5. Literature and Technical Background", { align: AlignmentType.LEFT }),
  p("6. System Requirements", { align: AlignmentType.LEFT }),
  p("7. System Architecture and Design", { align: AlignmentType.LEFT }),
  p("8. Detailed Module Design", { align: AlignmentType.LEFT }),
  p("9. Implementation", { align: AlignmentType.LEFT }),
  p("10. User Interface and Workflows", { align: AlignmentType.LEFT }),
  p("11. Evaluation Methodology and Metrics", { align: AlignmentType.LEFT }),
  p("12. Testing", { align: AlignmentType.LEFT }),
  p("13. Deployment", { align: AlignmentType.LEFT }),
  p("14. Results and Contributions", { align: AlignmentType.LEFT }),
  p("15. Discussion, Limitations, and Future Work", { align: AlignmentType.LEFT }),
  p("16. Conclusion", { align: AlignmentType.LEFT }),
  p("17. Defense Guide (Viva Voce Preparation)", { align: AlignmentType.LEFT }),
  p("References", { align: AlignmentType.LEFT }),
  p("Appendices", { align: AlignmentType.LEFT }),

  pageBreak(),

  // ========== 1 ==========
  h1("1. INTRODUCTION"),
  h2("1.1 Background"),
  p(
    "Undergraduate assessment in Computer Science in Nigeria typically combines continuous assessment with formal written examinations. Lecturers set theory questions that require discussion, definition, comparison, and explanation, often with explicit mark allocations and multi-part structure. Students prepare primarily from lecture notes, slides, and handouts distributed for each course. At the same time, generative AI tools have become widely available; students use them to generate practice questions. Without grounding in the actual course pack, such tools may produce plausible but syllabus-misaligned items—questions the student cannot answer from the notes and that the lecturer never intended to examine."
  ),
  p(
    "Retrieval-Augmented Generation (RAG) is a pattern that reduces hallucination by retrieving relevant passages from a corpus and conditioning the language model on those passages. StudyQuiz applies RAG to educational question generation: the corpus is the student’s own uploaded lecture material. The project is both an engineering system and a research artefact: it includes a baseline mode without retrieval so that grounded and ungrounded generation can be compared—the evaluation chapter contribution of the final-year work."
  ),
  h2("1.2 Motivation"),
  bullet("Local assessment culture uses theory papers (Discuss/Define/marks), not only multiple choice."),
  bullet("Students need practice items answerable from their notes, not generic internet knowledge."),
  bullet("Implementing BM25 from scratch makes the ranking mathematics defensible in oral examination."),
  bullet("Automatic quote-in-source checking provides a hard metric for grounding quality."),
  bullet("A working web demo supports project defense beyond slides alone."),

  h2("1.3 Project Overview"),
  p(
    "StudyQuiz is a web-based AI study assistant. The student uploads lecture material (PDF, Word, PowerPoint, or plain text, up to 20 MB). The system extracts text, chunks it, indexes chunks with BM25, generates practice assessments with Claude (structured outputs), and—for MCQ—grades answers server-side. For Exam Quiz, it produces a Nigerian-style theory paper and, after the student attempts answers offline, can generate marking points and model-answer outlines from the same notes."
  ),

  // ========== 2 ==========
  h1("2. PROBLEM STATEMENT"),
  p(
    "Existing general-purpose chatbots and generic quiz generators do not systematically ensure that practice questions are answerable from a specific lecture document. When questions are ungrounded, students waste revision time and may learn incorrect associations. Furthermore, local examination practice is dominated by multi-part theory questions with marks, yet most AI quiz tools only emit multiple-choice items. There is a need for a system that (a) grounds generation in the student’s materials via retrieval, (b) validates grounding with an automatic check, (c) supports both MCQ and theory-exam styles, and (d) exposes metrics and exports useful for academic evaluation of RAG versus baseline generation."
  ),

  // ========== 3 ==========
  h1("3. AIM AND OBJECTIVES"),
  h2("3.1 Aim"),
  p(
    "To design, implement, and evaluate a retrieval-augmented system that generates lecture-grounded practice assessments—multiple-choice and Nigerian-style theory examination papers—from student-uploaded course materials."
  ),
  h2("3.2 Specific Objectives"),
  num("To extract text from common lecture file formats and segment it into overlapping chunks suitable for retrieval."),
  num("To implement BM25 ranking from first principles and use it to select context for generation."),
  num("To generate structured multiple-choice questions with a large language model and grade them securely on the server."),
  num("To validate model-claimed source quotes against the document and compute quote-in-source rate for evaluation."),
  num("To generate theory examination papers in Nigerian BSc style and provide post-attempt answer guides from the notes."),
  num("To persist documents, quizzes, attempts, and evaluation rows, and to provide a usable web interface and automated tests."),

  // ========== 4 ==========
  h1("4. SCOPE AND LIMITATIONS"),
  h2("4.1 In Scope"),
  bullet("Text-based lecture documents: PDF, DOCX, PPTX, and plain text (max 20 MB)."),
  bullet("MCQ generation (1–15 questions) with four options and server-side scoring."),
  bullet("Exam Quiz theory papers (2–6 major questions) with parts, sub-parts, and marks."),
  bullet("Post-attempt marking guides and model-answer outlines from notes (not automatic essay scoring)."),
  bullet("RAG on/off (baseline) toggle for evaluation."),
  bullet("SQLite persistence, CSV evaluation export, unit tests, optional Vercel deploy."),
  h2("4.2 Out of Scope"),
  bullet("Optical character recognition (OCR) for scanned or handwritten notes."),
  bullet("Automatic grading of free-text essay answers (Exam Quiz is for practice and self-check)."),
  bullet("Multi-user authentication, roles, and multi-tenant SaaS."),
  bullet("Embedding-based vector search as the primary retriever (interface reserved as future upgrade)."),

  // ========== 5 ==========
  h1("5. LITERATURE AND TECHNICAL BACKGROUND"),
  h2("5.1 Retrieval-Augmented Generation"),
  p(
    "RAG combines information retrieval with generative models. Instead of relying only on parametric memory, the system retrieves external text and includes it in the prompt. In education, this improves alignment with a course corpus. StudyQuiz uses document-level RAG over a single uploaded lecture pack per session (extendable to multi-document sets as future work)."
  ),
  h2("5.2 BM25 Ranking"),
  p(
    "BM25 (Best Matching 25) is a probabilistic bag-of-words ranking function widely used in search engines (Robertson & Zaragoza, 2009). Term frequency saturates via parameter k1; document length is normalised via parameter b. Inverse document frequency up-weights rare terms. StudyQuiz implements BM25 directly in Python (k1 = 1.5, b = 0.75) so the formula can be presented and defended without treating a library as a black box. The retriever exposes index() and search() methods, allowing a future drop-in replacement with dense embeddings and a vector database (e.g. ChromaDB)."
  ),
  h2("5.3 Structured Outputs and Hallucination Control"),
  p(
    "Language models can produce free-form text that is hard to parse. StudyQuiz uses Anthropic’s structured output / parse API with Pydantic schemas so questions always include fields such as options, correct_index, and source_quote. Independently, grounding validation checks whether source_quote is a substring of the document (after normalisation). This two-layer approach—schema validity plus lexical grounding—supports the evaluation claim that RAG questions are more verifiably tied to source text than baseline questions."
  ),
  h2("5.4 Nigerian University Examination Format"),
  p(
    "Typical BSc written papers (e.g. FUOYE-style Computer Science examinations) use headers (course code, title, unit, time allowed), instructions such as “Answer Question One and any other…”, and major questions broken into (a)(b)(c) with roman (i)(ii)(iii) lists and mark allocations. Exam Quiz is deliberately designed to match this format rather than only Western-style MCQ banks."
  ),

  // ========== 6 ==========
  h1("6. SYSTEM REQUIREMENTS"),
  h2("6.1 Functional Requirements"),
  table(
    ["ID", "Requirement"],
    [
      ["FR1", "Upload and extract text from lecture files"],
      ["FR2", "Chunk text and build BM25 index"],
      ["FR3", "Generate MCQ quiz from document (RAG or baseline)"],
      ["FR4", "Grade MCQ attempts server-side"],
      ["FR5", "Validate source quotes; export evaluation CSV"],
      ["FR6", "Generate Nigerian-style theory exam paper"],
      ["FR7", "Generate post-attempt answers/marking guides from notes"],
      ["FR8", "Persist documents, quizzes, exams, attempts"],
      ["FR9", "Provide web UI for all primary workflows"],
    ],
    [1200, CONTENT_W - 1200]
  ),
  new Paragraph({ spacing: { after: 200 }, children: [] }),
  h2("6.2 Non-Functional Requirements"),
  bullet("Usability: clear steps, progress, timers, print/PDF for practice."),
  bullet("Reliability: grounding filter; parallel exam generation to reduce API 400s."),
  bullet("Testability: unit tests for BM25/chunking-independent modules without API keys."),
  bullet("Deployability: local uvicorn; optional Vercel serverless entry point."),
  bullet("Security of answers: MCQ correct options not sent to browser until submit."),

  // ========== 7 ==========
  h1("7. SYSTEM ARCHITECTURE AND DESIGN"),
  h2("7.1 High-Level Pipeline"),
  p(
    "The architecture maps to four stages after upload: (i) extraction and chunking (pdf_processor.py), (ii) retrieval (retriever.py + coverage.py), (iii) generation (generator.py / exam_generator.py via Claude), (iv) presentation and grading or answer-guide generation (main.py + static UI). Persistence is handled by store.py (SQLite)."
  ),
  p(
    "ASCII pipeline: Upload → Extract & Chunk → BM25 Index → Retrieve / Plan Coverage → LLM Generate (structured) → [MCQ: Ground-check → Serve → Grade] | [Exam: Serve paper → (later) Answer guides]."
  ),
  h2("7.2 Technology Stack"),
  table(
    ["Layer", "Technology"],
    [
      ["Language", "Python 3.x"],
      ["Web framework", "FastAPI + Uvicorn"],
      ["LLM", "Anthropic Claude (claude-haiku-4-5)"],
      ["Schemas", "Pydantic v2"],
      ["Documents", "pypdf, python-docx, python-pptx"],
      ["Database", "SQLite (stdlib sqlite3)"],
      ["Frontend", "Single HTML/CSS/JS page (no SPA framework)"],
      ["Tests", "pytest"],
      ["Deploy (optional)", "Vercel @vercel/python"],
      ["Version control", "GitHub (Handsomesinner/Studyquiz)"],
    ],
    [2800, CONTENT_W - 2800]
  ),
  new Paragraph({ spacing: { after: 200 }, children: [] }),
  h2("7.3 Design Principles"),
  num("Groundedness first: prefer verified questions over volume of questions."),
  num("Defensibility: BM25 and metrics can be explained mathematically in viva."),
  num("Separation of modes: MCQ auto-grade vs theory self-check—honest about essay grading limits."),
  num("Speed with quality: parallel per-question exam generation; deferred guides."),
  num("Replaceable retrieval: same index/search interface for future dense retrievers."),

  // ========== 8 ==========
  h1("8. DETAILED MODULE DESIGN"),
  h2("8.1 Document Processing (pdf_processor.py)"),
  p(
    "Text extraction routes by file extension: PDF via pypdf, DOCX via python-docx (paragraphs and tables), PPTX via python-pptx (shape text frames), otherwise UTF-8 plain text with a binary-guard heuristic. Whitespace is normalised. Chunking uses a sliding window of 200 words with 40-word overlap. Very short tails merge into the previous chunk. Overlap reduces loss of sentences that straddle boundaries."
  ),
  h2("8.2 BM25 Retriever (retriever.py)"),
  p(
    "Tokenisation lowercases text and extracts alphanumeric tokens, removing a small English stopword list. Indexing stores per-chunk term frequencies, document lengths, average length, and IDF. Search scores each chunk with the BM25 formula and returns the top-k (index, score) pairs. spread_sample() selects unique indices evenly from first to last chunk when no topic is given, improving long-document coverage versus always taking the opening pages."
  ),
  h2("8.3 Coverage Planning (coverage.py)"),
  p(
    "compute_top_k scales retrieval width with document size and number of questions (minimum 6, maximum 18). Topic queries use BM25 then diversify_indices (MMR-style minimum gap) so hits are not all adjacent windows. Long selections are batched (max 8 chunks per generation batch) for MCQ multi-section generation. allocate_questions distributes question counts across batches."
  ),
  h2("8.4 MCQ Generation (generator.py)"),
  p(
    "System prompt defines an examination setter persona. RAG mode requires questions answerable only from sources and a verbatim source_quote. Baseline mode uses title/topic knowledge and empty quotes. Difficulty (easy/medium/hard) appends guidance. Outputs are validated: exactly four options and correct_index in 0–3. Invalid items are filtered. Long documents may call generate_quiz_from_batches for section-wise generation."
  ),
  h2("8.5 Grounding Validation (grounding.py)"),
  p(
    "normalize_for_match applies Unicode NFKC, casefolding, quote/dash normalisation, and whitespace collapse. quote_in_source requires a minimum quote length and tries exact then punctuation-stripped containment. Each question receives grounded true/false and match_type (exact, normalized, not_found, empty_quote, baseline_empty). Quiz-level metrics include quote_in_source_rate and options_unique_rate. filter_grounded drops ungrounded items when require_grounding is enabled for RAG."
  ),
  h2("8.6 Persistence (store.py)"),
  p(
    "SQLite tables: documents, quizzes, quiz_attempts, eval_rows, exam_papers. JSON columns store chunks, questions, groundings, and full exam papers. BM25 is rebuilt from stored chunks on load. On Vercel/Lambda, the database path defaults to /tmp/studyquiz.db because the deployment filesystem is read-only."
  ),
  h2("8.7 Exam Generation (exam_generator.py)"),
  p(
    "Exam papers use nested Pydantic models: ExamPaper → ExamQuestion → ExamPart → ExamSubPart. To avoid long timeouts and HTTP 400 from oversized structured outputs, each major QUESTION is generated in a separate Claude call with a lean schema (no long guides). Calls run in parallel (thread pool). Document chunks are sliced by section (assign_chunk_slices) so each question sees a different region of the syllabus. After the student attempts the paper, fill_answer_guides retrieves BM25 context per question and generates guide_points, answer_outline, and source_quote for self-checking."
  ),
  h2("8.8 API Layer (main.py)"),
  p(
    "FastAPI exposes REST endpoints for documents, MCQ quiz lifecycle, evaluation export/summary, exam create/get/answers, and health. Startup initialises SQLite lazily-safe for serverless. Maximum upload size is 20 MB."
  ),

  // ========== 9 ==========
  h1("9. IMPLEMENTATION"),
  h2("9.1 Repository Structure"),
  table(
    ["Path", "Role"],
    [
      ["app/main.py", "FastAPI routes and request orchestration"],
      ["app/pdf_processor.py", "Extract + chunk"],
      ["app/retriever.py", "BM25"],
      ["app/coverage.py", "top_k, diversify, batches"],
      ["app/generator.py", "MCQ Claude generation"],
      ["app/grounding.py", "Quote validation + metrics"],
      ["app/exam_generator.py", "Theory paper + answer guides"],
      ["app/store.py", "SQLite persistence"],
      ["app/static/index.html", "Web UI"],
      ["api/index.py", "Vercel ASGI entry"],
      ["tests/", "Unit tests (no API key required for core tests)"],
      ["requirements.txt", "Dependencies"],
    ],
    [3200, CONTENT_W - 3200]
  ),
  new Paragraph({ spacing: { after: 200 }, children: [] }),
  h2("9.2 Approximate Code Scale"),
  p(
    "The application Python modules total on the order of three thousand lines including tests (order-of-magnitude at defense time), organised into cohesive modules rather than a single script—supporting maintainability arguments in viva."
  ),
  h2("9.3 Configuration"),
  bullet("ANTHROPIC_API_KEY — required for generation."),
  bullet("STUDYQUIZ_DB — optional path override for SQLite."),
  bullet("VERCEL / AWS_LAMBDA_FUNCTION_NAME — trigger /tmp database path."),

  // ========== 10 ==========
  h1("10. USER INTERFACE AND WORKFLOWS"),
  h2("10.1 Mode Selection"),
  p(
    "The header offers MCQ Quiz and Exam Quiz (theory paper). Mode switches labels, defaults (e.g. 5 MCQs vs 3 major exam questions), and which options appear (timer for MCQ; course code/title/time allowed for exam)."
  ),
  h2("10.2 Shared Step 1 — Upload"),
  p(
    "User selects a file or reuses a saved document from SQLite (survives local restarts). Status shows word and chunk counts after indexing."
  ),
  h2("10.3 MCQ Workflow"),
  num("Configure: number of questions, topic, difficulty, timer, RAG, require grounding."),
  num("Generate: retrieval plan → Claude → grounding → optional filter → quiz UI."),
  num("Take quiz: progress bar, unanswered warning, optional auto-submit on timer expiry."),
  num("Submit: server grades; show correct/wrong, explanations, quotes, badges."),
  num("Review wrong only; print; copy score; download evaluation CSV."),
  h2("10.4 Exam Quiz Workflow"),
  num("Configure course header fields and major question count."),
  num("Generate paper quickly via parallel per-question calls."),
  num("Student answers offline in a notebook (realistic exam practice)."),
  num("Generate answers from my notes: marking points + model outline per part."),
  num("Compare and print paper with guides."),
  h2("10.5 Why Exam Answers Are Not Auto-Scored"),
  p(
    "Free-text theory answers cannot be graded fairly with simple string match. Automatic essay scoring is out of scope and scientifically harder. StudyQuiz therefore supports self-assessment with material-grounded marking schemes—aligned with how human markers use schemes, and honest for defense."
  ),

  // ========== 11 ==========
  h1("11. EVALUATION METHODOLOGY AND METRICS"),
  h2("11.1 Research Comparison: RAG vs Baseline"),
  p(
    "For the same document, generate quizzes with use_rag=true and use_rag=false. Human raters (or the student and peers) can score each question for relevance to the course, factual correctness, and answerability from the source. Automatic metrics complement human ratings."
  ),
  h2("11.2 Automatic Metrics"),
  table(
    ["Metric", "Definition", "Interpretation"],
    [
      ["quote_in_source_rate", "Share of questions whose source_quote appears in the document", "Higher is better for RAG; baseline should be ~0 grounded"],
      ["options_unique_rate", "Share of MCQs with four distinct options", "Quality of option generation"],
      ["match_type", "exact / normalized / not_found / empty / baseline_empty", "Error analysis"],
      ["filtered_out", "Count dropped when require_grounding", "Cost of strict mode"],
    ],
    [2400, 3600, CONTENT_W - 6000]
  ),
  new Paragraph({ spacing: { after: 200 }, children: [] }),
  h2("11.3 Evaluation Tooling in Software"),
  bullet("GET /api/quiz/{id}/evaluation — per-question JSON."),
  bullet("GET /api/evaluation/summary — aggregate by RAG vs baseline."),
  bullet("GET /api/evaluation/export — CSV for Excel/graphs in the write-up."),
  bullet("pre_filter vs served phases — measure raw model grounding before dropping failures."),
  h2("11.4 Suggested Experimental Protocol for the Thesis"),
  num("Select N lecture documents across courses (e.g. security, OS, networks)."),
  num("For each document, generate equal-sized RAG and baseline MCQ sets."),
  num("Export CSV; compute quote-in-source rates automatically."),
  num("Have 2–3 raters score a sample on 1–5 scales; report means and inter-rater agreement if possible."),
  num("Optionally ablate top_k, difficulty, and require_grounding."),

  // ========== 12 ==========
  h1("12. TESTING"),
  p(
    "Automated tests use pytest and do not require an Anthropic API key for core logic. Test modules include: test_grounding.py (quote matching, metrics, filter), test_coverage.py (top_k scaling, diversification, batching, plans), test_store.py (SQLite save/load/reconnect), test_exam_generator.py (mark totals, chunk slice assignment, code/title guessing). At development freeze, the suite reported 27 passing tests. Generative API behaviour is validated manually in demos due to cost and non-determinism."
  ),

  // ========== 13 ==========
  h1("13. DEPLOYMENT"),
  h2("13.1 Local"),
  p("pip install -r requirements.txt; export ANTHROPIC_API_KEY; uvicorn app.main:app --reload; open http://127.0.0.1:8000."),
  h2("13.2 Vercel"),
  p(
    "api/index.py re-exports the FastAPI app; vercel.json routes all paths to the Python serverless function. SQLite uses /tmp (ephemeral). Cold starts and timeouts constrain very large exam jobs; parallel smaller calls mitigate this. GET /api/health verifies boot and database path."
  ),
  h2("13.3 GitHub"),
  p(
    "Source is maintained at https://github.com/Handsomesinner/Studyquiz under feature branches merged for features (grounding, persistence, coverage, UX, exam quiz, answer guides)."
  ),

  // ========== 14 ==========
  h1("14. RESULTS AND CONTRIBUTIONS"),
  h2("14.1 Engineering Deliverable"),
  bullet("End-to-end working system: upload → RAG quiz / theory paper → practice → metrics/export."),
  bullet("From-scratch BM25 suitable for oral mathematical explanation."),
  bullet("Grounding validation as an automatic quality gate and research metric."),
  bullet("Dual assessment modes matching Nigerian exam culture."),
  bullet("Persistence, tests, and deployment path for live demonstration."),
  h2("14.2 Research / Academic Contribution"),
  p(
    "The project’s research-facing contribution is the controlled comparison of grounded (RAG) versus ungrounded (baseline) question generation for lecture materials, supported by automatic quote-in-source measurement and optional human rating protocol—not merely “another ChatGPT wrapper.”"
  ),
  h2("14.3 Demonstrable Outcomes"),
  bullet("MCQ quizzes with verified source quotes and server-side scoring."),
  bullet("Theory papers resembling FUOYE-style CSC examinations (e.g. security topics)."),
  bullet("Self-check answer outlines after offline attempt."),
  bullet("CSV exports for chapter tables and graphs."),

  // ========== 15 ==========
  h1("15. DISCUSSION, LIMITATIONS, AND FUTURE WORK"),
  h2("15.1 Limitations"),
  bullet("No OCR: scanned PDFs fail extraction."),
  bullet("LLM non-determinism: questions vary between runs."),
  bullet("API cost and latency depend on Anthropic service and document size."),
  bullet("Vercel storage is ephemeral; multi-user production needs managed DB."),
  bullet("Theory answers are not auto-graded; guides may still omit edge cases if notes are sparse."),
  bullet("BM25 is lexical: synonym mismatch can reduce retrieval quality versus embeddings."),
  h2("15.2 Future Work"),
  bullet("Hybrid BM25 + dense embeddings with the same retriever interface."),
  bullet("Multi-document course packs and user accounts."),
  bullet("OCR pipeline for scanned handouts."),
  bullet("Spaced repetition of weak MCQ topics."),
  bullet("Richer evaluation harness (batch scripts, inter-rater UI)."),
  bullet("Optional human-in-the-loop editing of generated questions before practice."),

  // ========== 16 ==========
  h1("16. CONCLUSION"),
  p(
    "StudyQuiz demonstrates that retrieval-augmented generation, classical BM25 ranking implemented from first principles, structured LLM outputs, and automatic source-quote validation can be combined into a practical study system tailored to Nigerian university assessment formats. The dual modes—auto-graded MCQ and theory Exam Quiz with post-attempt material-grounded marking guides—reflect real student needs. Persistence, evaluation exports, tests, and a deployable web interface make the project suitable for demonstration and defense. Within stated scope limits, the system meets its aim: lecture-grounded practice that students can trust more than ungrounded chatbot quizzes."
  ),

  // ========== 17 ==========
  h1("17. DEFENSE GUIDE (VIVA VOCE PREPARATION)"),
  h2("17.1 One-Minute Elevator Pitch"),
  p(
    "“StudyQuiz turns my lecture notes into practice exams. It does not only ask Claude to invent questions—it retrieves the relevant parts of my PDF with BM25, generates questions from those parts, and checks that the model’s supporting quote is really in the document. I also support Nigerian-style theory papers, not only multiple choice.”"
  ),
  h2("17.2 Likely Questions and Strong Answers"),
  h3("Why RAG instead of only ChatGPT?"),
  p(
    "Parametric models alone may ignore the lecturer’s notes and invent syllabus content. RAG injects retrieved passages so questions track the pack the student will be examined on. Baseline mode exists specifically to measure that difference."
  ),
  h3("Why implement BM25 yourself?"),
  p(
    "To understand and defend the ranking formula (TF saturation, length normalisation, IDF), keep dependencies low, and keep a clean interface for upgrading to embeddings later without rewriting the whole system."
  ),
  h3("How do you know questions are grounded?"),
  p(
    "Each MCQ carries source_quote; grounding.py checks substring match after normalisation. Strict mode drops failures. quote_in_source_rate is logged for RAG vs baseline."
  ),
  h3("Why not auto-mark essay answers?"),
  p(
    "Fair automatic essay scoring is an open research problem and out of project scope. We generate marking schemes from the notes so students self-assess honestly—matching how human markers use schemes."
  ),
  h3("What if the PDF is scanned?"),
  p(
    "Out of scope: no OCR. System requires extractable text. That limitation is stated in the proposal and report."
  ),
  h3("Security of MCQ answers?"),
  p(
    "Correct indices stay on the server until submit; the client only receives questions and options during the attempt."
  ),
  h3("What is novel?"),
  p(
    "Combination of from-scratch BM25, grounding validation metric, RAG vs baseline evaluation tooling, and dual modes including local theory-exam format—integrated as a complete student-facing system for a final-year deliverable."
  ),
  h2("17.3 Demo Script (Recommended)"),
  num("Show GitHub repo and health endpoint if deployed."),
  num("Upload a known lecture (e.g. CSC 409 notes); show chunk count."),
  num("MCQ: RAG on, 5 questions, show grounding metrics and take quiz."),
  num("Optionally untick RAG and discuss baseline for evaluation."),
  num("Exam Quiz: generate paper; show QUESTION ONE structure with marks."),
  num("Generate answers from notes; show marking points under a part."),
  num("Show CSV export and mention automatic metrics."),
  h2("17.4 Ethical Considerations"),
  p(
    "The tool is for practice, not for cheating in live examinations. Generated content should not replace attendance or official past questions. API keys must not be committed to public repositories. Model outputs can be wrong; students should cross-check with lecturers’ materials."
  ),

  // ========== REFERENCES ==========
  h1("REFERENCES"),
  p("Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.", { align: AlignmentType.LEFT }),
  p("Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval.", { align: AlignmentType.LEFT }),
  p("Anthropic. Claude API documentation — Messages API and structured outputs. https://docs.anthropic.com/", { align: AlignmentType.LEFT }),
  p("FastAPI documentation. https://fastapi.tiangolo.com/", { align: AlignmentType.LEFT }),
  p("SQLite documentation. https://www.sqlite.org/docs.html", { align: AlignmentType.LEFT }),
  p("Federal University Oye-Ekiti — sample departmental examination paper formats (BSc Computer Science theory papers).", { align: AlignmentType.LEFT }),

  // ========== APPENDICES ==========
  h1("APPENDIX A — API SUMMARY"),
  table(
    ["Method", "Path", "Purpose"],
    [
      ["POST", "/api/documents", "Upload & index"],
      ["GET", "/api/documents", "List documents"],
      ["POST", "/api/quiz", "Create MCQ quiz"],
      ["POST", "/api/quiz/{id}/submit", "Grade MCQ"],
      ["GET", "/api/quiz/{id}/evaluation", "Grounding JSON"],
      ["GET", "/api/evaluation/summary", "RAG vs baseline rates"],
      ["GET", "/api/evaluation/export", "CSV export"],
      ["POST", "/api/exam", "Create theory paper"],
      ["GET", "/api/exam/{id}", "Fetch exam"],
      ["POST", "/api/exam/{id}/answers", "Marking guides from notes"],
      ["GET", "/api/health", "Health / DB path"],
    ],
    [1400, 3600, CONTENT_W - 5000]
  ),
  new Paragraph({ spacing: { after: 280 }, children: [] }),

  h1("APPENDIX B — BM25 FORMULA (DEFENSE AID)"),
  p(
    "For query terms q and document (chunk) D, a common BM25 score form is the sum over query terms of IDF(t) × (tf × (k1+1)) / (tf + k1 × (1 − b + b × |D|/avgdl)), where tf is term frequency in D, |D| is document length, avgdl is average document length, k1 controls TF saturation (1.5 in StudyQuiz), and b controls length normalisation (0.75). IDF uses a smoothed form log((N − df + 0.5)/(df + 0.5) + 1) with N the number of chunks and df the document frequency of term t."
  ),

  h1("APPENDIX C — SYSTEM LIMITS"),
  table(
    ["Limit", "Value"],
    [
      ["Max upload", "20 MB"],
      ["MCQ count", "1–15"],
      ["Exam major questions", "2–6"],
      ["Chunk size", "200 words, 40 overlap"],
      ["MCQ retrieval top_k", "up to 18"],
      ["Exam context per question", "6 chunks × ~110 words"],
      ["Vercel DB", "/tmp (ephemeral)"],
    ],
    [4000, CONTENT_W - 4000]
  ),
  new Paragraph({ spacing: { after: 280 }, children: [] }),

  h1("APPENDIX D — GLOSSARY"),
  bullet("RAG — Retrieval-Augmented Generation."),
  bullet("BM25 — Probabilistic bag-of-words ranking function."),
  bullet("Grounding — Evidence that a claim (quote/answer) is supported by source text."),
  bullet("Baseline — Generation without retrieved lecture context."),
  bullet("Structured output — LLM response forced into a schema (JSON/Pydantic)."),
  bullet("Chunk — Segment of a document used as a retrieval unit."),

  new Paragraph({ spacing: { before: 400 }, children: [] }),
  center("— End of Report —", { italics: true, size: 22 }),
  center("StudyQuiz · Famule Oluwapamilerin Solomon · FTP/CSC/26/0133938", { size: 18 }),
];

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "Times New Roman", size: 24 },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
      {
        reference: "numbers",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              children: [
                new TextRun({
                  text: "StudyQuiz — Project Defense Report",
                  font: "Times New Roman",
                  size: 18,
                  italics: true,
                  color: "666666",
                }),
              ],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "Page ", font: "Times New Roman", size: 18 }),
                new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 18 }),
                new TextRun({
                  text: "  |  Famule O. S.  |  FTP/CSC/26/0133938",
                  font: "Times New Roman",
                  size: 18,
                }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

const out = path.join(__dirname, "StudyQuiz_Project_Defense_Report.docx");
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(out, buffer);
  console.log("Wrote", out);
});
