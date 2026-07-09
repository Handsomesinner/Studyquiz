"""Question generation: stage (iii) of the RAG pipeline.

Retrieved chunks are placed into the prompt and Claude is asked to generate
multiple-choice questions grounded in that material. Structured outputs
(a Pydantic schema enforced by the API) guarantee the response is valid,
parseable JSON — no fragile string parsing.

Setting `use_rag=False` skips the retrieved context entirely. That is the
baseline condition for the project's evaluation chapter: comparing the
quality of grounded (RAG) vs ungrounded (plain LLM) questions.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

MODEL = "claude-haiku-4-5"


class QuizQuestion(BaseModel):
    question: str
    options: list[str]  # exactly 4 answer options
    correct_index: int  # 0-3, index into options
    explanation: str
    source_quote: str  # short quote from the material that supports the answer ("" in baseline mode)


class Quiz(BaseModel):
    questions: list[QuizQuestion]


class GenerationError(Exception):
    """Raised with a user-presentable message when generation fails."""


def _client():
    # Lazy import so unit tests (e.g. grounding) don't need the SDK installed.
    import anthropic

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise GenerationError(
            "No Anthropic API key configured. Set the ANTHROPIC_API_KEY "
            "environment variable before starting the server "
            "(get a key at https://platform.claude.com/)."
        )
    return anthropic.Anthropic()


SYSTEM_PROMPT = (
    "You are an examination question setter for a university course. "
    "You write clear, unambiguous multiple-choice questions with exactly four "
    "options each: one correct answer and three plausible distractors. "
    "Distractors must be wrong but believable — common misconceptions work well. "
    "Vary which option position holds the correct answer."
)


def generate_quiz(
    *,
    num_questions: int,
    doc_title: str,
    context_chunks: list[str],
    topic: str | None,
    use_rag: bool,
) -> Quiz:
    if use_rag:
        sources = "\n\n".join(
            f"[Source {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        task = (
            f"Generate exactly {num_questions} multiple-choice questions based "
            f"STRICTLY on the lecture material below. Every question must be "
            f"answerable from the sources — do not use outside knowledge. For "
            f"each question, set source_quote to a short verbatim phrase from "
            f"the source that supports the correct answer.\n\n"
            f"Lecture material from \"{doc_title}\":\n\n{sources}"
        )
    else:
        # Baseline condition: no retrieved context, title/topic only.
        task = (
            f"Generate exactly {num_questions} multiple-choice questions for a "
            f"university course document titled \"{doc_title}\". Use your own "
            f"knowledge of the subject. Set source_quote to an empty string "
            f"for every question."
        )

    if topic:
        task += f"\n\nFocus the questions on this topic: {topic}"

    import anthropic

    client = _client()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task}],
            output_format=Quiz,
        )
    except anthropic.AuthenticationError:
        raise GenerationError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise GenerationError("Rate limited by the Claude API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        raise GenerationError(f"Claude API error ({e.status_code}). Try again shortly.")
    except anthropic.APIConnectionError:
        raise GenerationError("Could not reach the Claude API. Check your internet connection.")

    quiz = response.parsed_output
    if quiz is None:
        raise GenerationError("The model returned an unparseable quiz. Try again.")

    # The JSON schema can't express "exactly 4 options" (array-length
    # constraints aren't supported), so enforce it here.
    quiz.questions = [
        q for q in quiz.questions if len(q.options) == 4 and 0 <= q.correct_index <= 3
    ]
    if not quiz.questions:
        raise GenerationError("No valid questions were generated. Try again.")
    return quiz
