"""Exam Quiz generation — Nigerian university theory-paper style.

Unlike the MCQ study quiz, this mode produces long-form examination questions
in the style of BSc papers (e.g. FUOYE CSC-style): QUESTION ONE / TWO with
parts (a)(b)(c), roman sub-parts (i)(ii)(iii), and mark allocations.
Students use it to *prepare* for written exams, not for auto-scored MCQs.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from .generator import GenerationError, MODEL, _normalize_difficulty

# Re-export for callers that only import this module.
__all__ = [
    "ExamPaper",
    "ExamQuestion",
    "ExamPart",
    "ExamSubPart",
    "generate_exam_paper",
    "GenerationError",
]


class ExamSubPart(BaseModel):
    label: str  # i, ii, iii
    text: str


class ExamPart(BaseModel):
    label: str  # a, b, c, …
    prompt: str
    marks: float
    subparts: list[ExamSubPart] = Field(default_factory=list)
    # Prep aid: bullet points a strong answer should cover (not shown until revealed).
    guide_points: list[str] = Field(default_factory=list)
    source_quote: str = ""  # short support from material when RAG is on


class ExamQuestion(BaseModel):
    number: int  # 1, 2, 3, …
    heading: str  # e.g. "QUESTION ONE"
    parts: list[ExamPart]


class ExamPaper(BaseModel):
    institution: str = "StudyQuiz Exam Prep"
    faculty: str = "Faculty of Science"
    department: str = "Department of Computer Science"
    exam_title: str = "BSc. Degree Examination (Practice Paper)"
    session: str = "Practice Session"
    course_code: str = "CSC XXX"
    course_title: str = "Course Title"
    course_unit: str = "2"
    time_allowed: str = "2 Hrs."
    instructions: list[str]
    questions: list[ExamQuestion]


EXAM_SYSTEM_PROMPT = (
    "You are a chief examiner setting a Nigerian university undergraduate "
    "Computer Science written examination (BSc level), in the style used by "
    "institutions such as Federal University Oye-Ekiti (FUOYE) and similar "
    "Nigerian universities.\n\n"
    "Paper conventions you MUST follow:\n"
    "- Long-form theory questions, NOT multiple choice.\n"
    "- Number major questions as QUESTION ONE, QUESTION TWO, … "
    "(set heading accordingly; number is 1, 2, 3…).\n"
    "- Break each major question into lettered parts (a), (b), (c), …\n"
    "- Use roman numerals (i), (ii), (iii) for lists inside a part when the "
    "student must discuss several items.\n"
    "- Assign marks per part (e.g. 6 Marks, 3 Marks). Marks should be "
    "realistic and sum sensibly (typical part 2–8 marks).\n"
    "- Verbs: Discuss, Define, Explain, Describe, List and explain, "
    "Compare, Give a brief discussion, With suitable examples…\n"
    "- Compulsory-style Question One should be broader; later questions "
    "can be more focused sections of the syllabus.\n"
    "- Provide guide_points: 3–6 short bullets a good student answer should "
    "mention (for revision, not wording to copy).\n"
    "- When source material is provided, every part must be answerable from "
    "that material; set source_quote to a short verbatim phrase from it. "
    "If no material, leave source_quote empty.\n"
    "- Do NOT invent university names different from what the user asks; "
    "use the header fields the user requests."
)

EXAM_DIFFICULTY = {
    "easy": (
        "Difficulty EASY: favour Define / List / State / Briefly explain. "
        "Fewer multi-part cascades; marks mostly 2–5."
    ),
    "medium": (
        "Difficulty MEDIUM: mix Discuss / Explain with examples and short "
        "comparisons. Parts often 3–6 marks."
    ),
    "hard": (
        "Difficulty HARD: deeper Discuss / Critically examine / Compare and "
        "contrast / design defence mechanisms. Multi-item roman lists and "
        "higher mark parts (4–8)."
    ),
}


def _client():
    import anthropic

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise GenerationError(
            "No Anthropic API key configured. Set the ANTHROPIC_API_KEY "
            "environment variable before starting the server "
            "(get a key at https://platform.claude.com/)."
        )
    return anthropic.Anthropic()


def generate_exam_paper(
    *,
    num_questions: int,
    doc_title: str,
    context_chunks: list[str],
    topic: str | None,
    use_rag: bool,
    difficulty: str = "medium",
    course_code: str | None = None,
    course_title: str | None = None,
    time_allowed: str = "2 Hrs.",
) -> ExamPaper:
    """Generate a full theory exam paper grounded (optionally) in lecture notes."""
    num_questions = max(2, min(num_questions, 6))
    diff = _normalize_difficulty(difficulty)

    code = (course_code or "").strip() or _guess_code(doc_title)
    title = (course_title or "").strip() or _guess_title(doc_title)

    if use_rag and context_chunks:
        sources = "\n\n".join(
            f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        material_block = (
            f"Base ALL questions STRICTLY on this lecture material from "
            f"\"{doc_title}\". Do not examine topics absent from the sources.\n\n"
            f"{sources}"
        )
    else:
        material_block = (
            f"No lecture extract provided. Set a realistic paper for a university "
            f"course titled \"{doc_title}\" / \"{title}\" using standard syllabus "
            f"knowledge for that subject. Leave every source_quote empty."
        )

    topic_line = f"\nEmphasise this theme where possible: {topic}." if topic else ""

    task = f"""Set a practice BSc written examination paper.

HEADER FIELDS (use these exact values in the JSON):
- institution: "StudyQuiz Exam Prep"
- faculty: "Faculty of Science"
- department: "Department of Computer Science"
- exam_title: "BSc. Degree Examination (Practice Paper)"
- session: "Practice Session"
- course_code: "{code}"
- course_title: "{title}"
- course_unit: "2"
- time_allowed: "{time_allowed}"

INSTRUCTIONS (include these, adapt marks language if needed):
1. Answer Question One and Any Other {max(1, num_questions - 1)} Question(s).
2. No jotting is allowed on the Question Paper. (practice note: write in your notebook)
3. Credit will be given for clarity, structure, and relevant examples.

STRUCTURE:
- Produce exactly {num_questions} major questions (QUESTION ONE …).
- Question One should have more parts (about 5–7 parts) like a compulsory question.
- Other questions: about 3–5 parts each.
- Use roman sub-parts when asking students to cover several named items.
- Total marks across the paper should feel like a real exam (roughly 40–70 if summing all parts shown).

{EXAM_DIFFICULTY[diff]}
{topic_line}

MATERIAL:
{material_block}
"""

    import anthropic

    client = _client()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=EXAM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task}],
            output_format=ExamPaper,
        )
    except anthropic.AuthenticationError:
        raise GenerationError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise GenerationError("Rate limited by the Claude API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        raise GenerationError(f"Claude API error ({e.status_code}). Try again shortly.")
    except anthropic.APIConnectionError:
        raise GenerationError("Could not reach the Claude API. Check your internet connection.")

    paper = response.parsed_output
    if paper is None or not paper.questions:
        raise GenerationError("The model returned an unparseable exam paper. Try again.")

    # Normalise headings / numbers
    ordinals = [
        "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
    ]
    cleaned: list[ExamQuestion] = []
    for i, q in enumerate(paper.questions[:num_questions]):
        parts = [p for p in q.parts if p.prompt.strip() and p.marks > 0]
        if not parts:
            continue
        num = i + 1
        heading = q.heading.strip() if q.heading else f"QUESTION {ordinals[i] if i < len(ordinals) else num}"
        if not heading.upper().startswith("QUESTION"):
            heading = f"QUESTION {ordinals[i] if i < len(ordinals) else num}"
        cleaned.append(
            ExamQuestion(number=num, heading=heading.upper(), parts=parts)
        )

    if not cleaned:
        raise GenerationError("No valid exam questions were generated. Try again.")

    paper.questions = cleaned
    paper.course_code = code
    paper.course_title = title
    paper.time_allowed = time_allowed or paper.time_allowed
    if not paper.instructions:
        paper.instructions = [
            f"Answer Question One and Any Other {max(1, len(cleaned) - 1)} Question(s).",
            "No jotting is allowed on the Question Paper.",
        ]
    return paper


def _guess_code(doc_title: str) -> str:
    import re

    m = re.search(r"\b([A-Z]{2,4}\s*\d{2,3})\b", doc_title or "", re.I)
    if m:
        return m.group(1).upper().replace("  ", " ")
    return "CSC 000"


def _guess_title(doc_title: str) -> str:
    name = (doc_title or "Course Materials").rsplit(".", 1)[0]
    name = name.replace("_", " ").replace("-", " ").strip()
    return name[:80] if name else "Course Materials"


def paper_total_marks(paper: ExamPaper) -> float:
    return sum(p.marks for q in paper.questions for p in q.parts)
