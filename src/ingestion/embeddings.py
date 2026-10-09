"""Create Hugging Face embeddings for LangChain document chunks."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from src.ingestion.splitter import load_and_split_documents

logger = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _configure_logging() -> Path:
    """Write embedding activity to ``logs/embeddings.log``."""

    global _LOGGER_CONFIGURED
    log_path = Path(__file__).resolve().parents[2] / "logs" / "embeddings.log"
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


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Create the LangChain Hugging Face embedding model.

    ``HF_TOKEN`` is read from the project's ``.env`` file when present and is
    never included in logs. Public models can also work without a token.
    """

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    token = os.getenv("HF_TOKEN")
    model_kwargs = {"token": token} if token else {}

    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Initializing Hugging Face embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_documents(
    documents: list[Document],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[list[float]]:
    """Create one embedding vector for each chunk document."""

    log_path = _configure_logging()
    if not documents:
        logger.warning("No documents supplied for embedding")
        return []

    embedding_model = get_embedding_model(model_name)
    texts = [document.page_content for document in documents]
    vectors = embedding_model.embed_documents(texts)
    dimensions = len(vectors[0]) if vectors else 0
    logger.info(
        "Created embeddings: documents=%d dimensions=%d model=%s log_file=%s",
        len(vectors),
        dimensions,
        model_name,
        log_path,
    )
    return vectors


def load_split_and_embed_documents(
    data_dir: str | Path | None = None,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strict: bool = False,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[list[Document], list[list[float]]]:
    """Load files, split them into chunks, and create their embeddings."""

    documents = load_and_split_documents(
        data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strict=strict,
    )
    vectors = embed_documents(documents, model_name=model_name)
    return documents, vectors


if __name__ == "__main__":
    chunks, vectors = load_split_and_embed_documents()
    dimensions = len(vectors[0]) if vectors else 0
    print(
        f"Created {len(vectors)} embedding vector(s) with {dimensions} dimensions. "
        "See logs/embeddings.log for details."
    )


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "embed_documents",
    "get_embedding_model",
    "load_split_and_embed_documents",
]