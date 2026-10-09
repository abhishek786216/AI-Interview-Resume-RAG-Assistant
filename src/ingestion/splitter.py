"""Split LangChain documents into searchable chunks."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import load_documents

logger = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False


def _configure_logging() -> Path:
    """Write splitter activity to ``logs/splitter.log``."""

    global _LOGGER_CONFIGURED
    log_path = Path(__file__).resolve().parents[2] / "logs" / "splitter.log"
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


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split loaded documents and add chunk metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")

    log_path = _configure_logging()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    logger.info(
        "Starting split: documents=%d chunk_size=%d chunk_overlap=%d",
        len(documents),
        chunk_size,
        chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    source_counts: defaultdict[str, int] = defaultdict(int)
    source_sizes: defaultdict[str, list[int]] = defaultdict(list)
    for chunk in chunks:
        source = str(
            chunk.metadata.get("relative_source", chunk.metadata.get("source", "unknown"))
        )
        source_counts[source] += 1
        source_sizes[source].append(len(chunk.page_content))

    source_chunk_indexes: defaultdict[str, int] = defaultdict(int)
    for chunk in chunks:
        source = str(
            chunk.metadata.get("relative_source", chunk.metadata.get("source", "unknown"))
        )
        chunk.metadata["chunk_index"] = source_chunk_indexes[source]
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        source_chunk_indexes[source] += 1

    for source in sorted(source_counts):
        sizes = source_sizes[source]
        logger.info(
            "Split %s into %d chunk(s), characters=%d-%d",
            source,
            source_counts[source],
            min(sizes),
            max(sizes),
        )
    logger.info(
        "Completed split: input_documents=%d output_chunks=%d log_file=%s",
        len(documents),
        len(chunks),
        log_path,
    )
    return chunks


def load_and_split_documents(
    data_dir: str | Path | None = None,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strict: bool = False,
) -> list[Document]:
    """Load source files with LangChain and then split them into chunks."""

    documents = load_documents(data_dir, strict=strict)
    return split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


if __name__ == "__main__":
    chunks = load_and_split_documents()
    print(f"Created {len(chunks)} chunk(s). See logs/splitter.log for details.")


__all__ = ["load_and_split_documents", "split_documents"]