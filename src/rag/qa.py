"""Answer questions with retrieved context and a Groq-hosted chat model."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# When this file is run directly, Python starts with ``src/rag`` on
# sys.path instead of the project root, so the absolute ``src`` imports need
# the root added explicitly.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retrieval.retriever import retrieve_documents

logger = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False

DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"
DEFAULT_RETRIEVAL_K = 3

_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You answer questions using only the provided context.
If the answer is not present in the context, say:
"I do not have enough information in the loaded documents."
If the context directly names a person, company, project, technology, or date,
answer that fact directly. Do not invent facts. Keep the answer clear and concise.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def _configure_logging() -> Path:
    """Write question-answer activity to ``logs/qa.log``."""

    global _LOGGER_CONFIGURED
    log_path = Path(__file__).resolve().parents[2] / "logs" / "qa.log"
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


def get_llm(model_name: str | None = None):
    """Create the Groq chat model using the project's ``GROQ_API_KEY``."""

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add your Groq API key to the project's .env file."
        )

    from langchain_groq import ChatGroq

    project_env = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    selected_model = (model_name or project_env.get("GROQ_MODEL") or DEFAULT_LLM_MODEL).strip()
    logger.info("Initializing Groq chat model: %s", selected_model)
    return ChatGroq(
        model=selected_model,
        temperature=0,
        api_key=api_key,
    )


def _format_context(documents: list[Document]) -> str:
    """Format retrieved chunks with source labels for the prompt."""

    sections = []
    for rank, document in enumerate(documents, start=1):
        source = document.metadata.get(
            "relative_source",
            document.metadata.get("source", "unknown"),
        )
        chunk_index = document.metadata.get("chunk_index", "unknown")
        sections.append(
            f"[Source {rank}: {source}, chunk {chunk_index}]\n"
            f"{document.page_content}"
        )
    return "\n\n".join(sections)


def answer_question(
    question: str,
    *,
    k: int = DEFAULT_RETRIEVAL_K,
    model_name: str | None = None,
) -> tuple[str, list[Document]]:
    """Retrieve context and answer one question with the Groq LLM."""

    _configure_logging()
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than zero")

    documents = retrieve_documents(question, k=k)
    if not documents:
        raise RuntimeError("No documents were retrieved for this question")

    prompt = _QA_PROMPT.invoke(
        {"context": _format_context(documents), "question": question}
    )
    project_env = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    selected_model = (
        model_name
        or project_env.get("GROQ_MODEL")
        or DEFAULT_LLM_MODEL
    ).strip()
    try:
        response = get_llm(selected_model).invoke(prompt)
    except (ImportError, RuntimeError) as error:
        logger.exception("Groq request failed: model=%s", selected_model)
        raise RuntimeError(
            f"Groq request failed for model '{selected_model}'. "
            "Check GROQ_MODEL and your Groq model access."
        ) from error
    except Exception as error:
        # The Groq SDK exposes several API error subclasses across versions.
        if error.__class__.__module__.startswith("groq."):
            logger.exception("Groq request failed: model=%s", selected_model)
            raise RuntimeError(
                f"Groq request failed for model '{selected_model}'. "
                "Check GROQ_MODEL and your Groq model access."
            ) from error
        raise
    answer = response.content
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The LLM returned an empty answer")

    logger.info(
        "Answered question: retrieved=%d model=%s question=%r",
        len(documents),
        selected_model,
        question,
    )
    return answer.strip(), documents


if __name__ == "__main__":
    print("Ask a question about your loaded documents. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        try:
            answer, sources = answer_question(question)
            print(f"\nAnswer:\n{answer}")
            print("\nSources:")
            for source in sources:
                print(
                    f"- {source.metadata.get('relative_source', 'unknown')} "
                    f"(chunk {source.metadata.get('chunk_index', 'unknown')})"
                )
        except (RuntimeError, ValueError) as error:
            print(f"\nError: {error}")


__all__ = ["answer_question", "get_llm"]