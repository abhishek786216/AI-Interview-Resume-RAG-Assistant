"""Retrieve the most relevant chunks from the local FAISS index."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.embeddings import DEFAULT_EMBEDDING_MODEL
from src.retrieval.vectorstore import load_vector_store

logger = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False
_DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "storage" / "faiss_index"


def _configure_logging() -> Path:
    """Write retrieval activity to ``logs/retriever.log``."""

    global _LOGGER_CONFIGURED
    log_path = Path(__file__).resolve().parents[2] / "logs" / "retriever.log"
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


def retrieve_documents(
    query: str,
    *,
    k: int = 3,
    index_dir: str | Path = _DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[Document]:
    """Return the top ``k`` most relevant chunks for ``query``."""

    _configure_logging()
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than zero")

    store = load_vector_store(index_dir, model_name=model_name)
    results = store.similarity_search(query, k=k)
    logger.info(
        "Retrieved documents: query=%r requested_k=%d returned=%d index=%s",
        query,
        k,
        len(results),
        Path(index_dir).expanduser().resolve(),
    )
    for rank, document in enumerate(results, start=1):
        logger.info(
            "Result rank=%d source=%s chunk_index=%s chunk_size=%d",
            rank,
            document.metadata.get("relative_source", document.metadata.get("source", "unknown")),
            document.metadata.get("chunk_index", "unknown"),
            len(document.page_content),
        )
    return results


if __name__ == "__main__":
    query = input("Enter your query: ").strip()
    results = retrieve_documents(query)
    print(f"\nTop {len(results)} result(s):")
    for rank, document in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {document.metadata.get('relative_source', 'unknown')}")
        print(f"Chunk: {document.metadata.get('chunk_index', 'unknown')}")
        print(document.page_content)
    print("\nSee logs/retriever.log for retrieval details.")


__all__ = ["retrieve_documents"]