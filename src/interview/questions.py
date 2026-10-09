"""Generate interview questions from retrieved resume and job context."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.rag.qa import DEFAULT_LLM_MODEL, get_llm
from src.retrieval.retriever import retrieve_documents

logger = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False


@dataclass(frozen=True)
class InterviewQuestion:
    """An interview question and the documents used to generate it."""

    question: str
    sources: list[Document]


_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced technical interviewer.
Generate exactly one interview question using only the provided context.
Prefer a specific question about a project, technology, design decision,
implementation trade-off, or measurable result. Do not invent a project or
technology that is absent from the context. Return only the question text.

Context:
{context}""",
        ),
        ("human", "Generate one interview question for this focus: {focus}"),
    ]
)


def _configure_logging() -> Path:
    """Write question-generation activity to ``logs/questions.log``."""

    global _LOGGER_CONFIGURED
    log_path = Path(__file__).resolve().parents[2] / "logs" / "questions.log"
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
    sections = []
    for rank, document in enumerate(documents, start=1):
        source = document.metadata.get(
            "relative_source", document.metadata.get("source", "unknown")
        )
        sections.append(f"[Source {rank}: {source}]\n{document.page_content}")
    return "\n\n".join(sections)


def generate_interview_question(
    focus: str = "the candidate's projects and technical decisions",
    *,
    k: int = 5,
    model_name: str | None = None,
) -> InterviewQuestion:
    """Retrieve context and generate one grounded interview question."""

    _configure_logging()
    focus = focus.strip()
    if not focus:
        raise ValueError("focus must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than zero")

    documents = retrieve_documents(focus, k=k)
    if not documents:
        raise RuntimeError("No documents were retrieved for question generation")

    response = get_llm(model_name).invoke(
        _QUESTION_PROMPT.invoke(
            {"context": _format_context(documents), "focus": focus}
        )
    )
    question = response.content.strip()
    if not question:
        raise RuntimeError("The LLM returned an empty interview question")

    logger.info(
        "Generated interview question: focus=%r sources=%d model=%s",
        focus,
        len(documents),
        model_name or DEFAULT_LLM_MODEL,
    )
    return InterviewQuestion(question=question, sources=documents)


if __name__ == "__main__":
    generated = generate_interview_question()
    print(generated.question)
    print("\nSources:")
    for source in generated.sources:
        print(f"- {source.metadata.get('relative_source', 'unknown')}")


__all__ = ["InterviewQuestion", "generate_interview_question"]