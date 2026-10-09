"""Evaluate interview answers against retrieved candidate context."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.rag.qa import DEFAULT_LLM_MODEL, get_llm

logger = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False


@dataclass(frozen=True)
class AnswerEvaluation:
    """Structured feedback for one interview answer."""

    score: int
    correct_points: list[str]
    missing_points: list[str]
    feedback: str


_EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a fair technical interviewer evaluating an answer.
Use only the supplied context and question. Return valid JSON only, with this
exact shape:
{{"score": 0, "correct_points": [], "missing_points": [], "feedback": ""}}

The score must be an integer from 0 to 10. `correct_points` and
`missing_points` must be arrays of short strings. Do not invent facts absent
from the context. If the context does not contain enough expected detail,
explain that in `missing_points` and `feedback`.

Context:
{context}""",
        ),
        (
            "human",
            "Question:\n{question}\n\nCandidate answer:\n{answer}",
        ),
    ]
)


def _configure_logging() -> Path:
    """Write evaluation activity to ``logs/evaluator.log``."""

    global _LOGGER_CONFIGURED
    log_path = Path(__file__).resolve().parents[2] / "logs" / "evaluator.log"
    if not _LOGGER_CONFIGURED:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _LOGGER_CONFIGURED = True
    return log_path


def _format_context(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


def evaluate_answer(
    question: str,
    answer: str,
    context: list[Document],
    *,
    model_name: str | None = None,
) -> AnswerEvaluation:
    """Evaluate an answer and return validated structured feedback."""

    _configure_logging()
    question = question.strip()
    answer = answer.strip()
    if not question:
        raise ValueError("question must not be empty")
    if not answer:
        raise ValueError("answer must not be empty")
    if not context:
        raise ValueError("context must not be empty")

    response = get_llm(model_name).invoke(
        _EVALUATION_PROMPT.invoke(
            {
                "context": _format_context(context),
                "question": question,
                "answer": answer,
            }
        )
    )
    try:
        payload = json.loads(response.content)
        score = int(payload["score"])
        correct_points = [str(item) for item in payload["correct_points"]]
        missing_points = [str(item) for item in payload["missing_points"]]
        feedback = str(payload["feedback"]).strip()
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        logger.exception("The LLM returned invalid evaluation JSON")
        raise RuntimeError("The LLM returned invalid evaluation JSON") from error

    if not 0 <= score <= 10:
        raise RuntimeError(f"Evaluation score must be between 0 and 10, got {score}")
    if not feedback:
        raise RuntimeError("Evaluation feedback must not be empty")

    logger.info(
        "Evaluated answer: score=%d correct=%d missing=%d model=%s",
        score,
        len(correct_points),
        len(missing_points),
        model_name or DEFAULT_LLM_MODEL,
    )
    return AnswerEvaluation(score, correct_points, missing_points, feedback)


__all__ = ["AnswerEvaluation", "evaluate_answer"]