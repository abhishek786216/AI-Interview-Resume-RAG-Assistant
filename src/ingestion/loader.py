"""Load source documents with LangChain document loaders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document

logger = logging.getLogger(__name__)
_LOGGER_CONFIGURED = False

_TEXT_EXTENSIONS = {".css", ".ini", ".md", ".rst", ".text", ".txt", ".xml", ".yaml", ".yml", ".json"}
_LOADER_FACTORIES: dict[str, Callable[[Path], Any]] = {}


def _text_loader(path: Path) -> Any:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding="utf-8-sig")


def _csv_loader(path: Path) -> Any:
    from langchain_community.document_loaders import CSVLoader

    return CSVLoader(str(path), encoding="utf-8-sig")


def _html_loader(path: Path) -> Any:
    from langchain_community.document_loaders import BSHTMLLoader

    return BSHTMLLoader(str(path), open_encoding="utf-8-sig")


def _pdf_loader(path: Path) -> Any:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path))


def _docx_loader(path: Path) -> Any:
    from langchain_community.document_loaders import Docx2txtLoader

    return Docx2txtLoader(str(path))


def _pptx_loader(path: Path) -> Any:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader

    return UnstructuredPowerPointLoader(str(path))


def _xlsx_loader(path: Path) -> Any:
    from langchain_community.document_loaders import UnstructuredExcelLoader

    return UnstructuredExcelLoader(str(path), mode="elements")


_LOADER_FACTORIES.update(
    {
        **{extension: _text_loader for extension in _TEXT_EXTENSIONS},
        ".csv": _csv_loader,
        ".html": _html_loader,
        ".htm": _html_loader,
        ".pdf": _pdf_loader,
        ".docx": _docx_loader,
        ".pptx": _pptx_loader,
        ".xlsx": _xlsx_loader,
    }
)


def _default_data_dir() -> Path:
    """Return ``data/`` relative to the project root, not the process cwd."""

    return Path(__file__).resolve().parents[2] / "data"


def _configure_logging() -> Path:
    """Write loader activity to ``logs/loader.log`` without duplicate handlers."""

    global _LOGGER_CONFIGURED
    log_path = Path(__file__).resolve().parents[2] / "logs" / "loader.log"
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


def _add_metadata(documents: list[Document], path: Path, root: Path) -> list[Document]:
    """Add consistent project metadata to every LangChain document."""

    relative_source = path.relative_to(root)
    category = relative_source.parts[0] if len(relative_source.parts) > 1 else root.name
    metadata = {
        "source": str(path),
        "relative_source": relative_source.as_posix(),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "category": category,
    }
    for document in documents:
        document.metadata.update(metadata)
    return documents


def load_documents(
    data_dir: str | Path | None = None,
    *,
    strict: bool = False,
) -> list[Document]:
    """Load supported files recursively as LangChain ``Document`` objects.

    A file can produce more than one document when its LangChain loader returns
    pages, rows, slides, or spreadsheet elements. Unsupported and empty files
    are skipped. With ``strict=False``, one failed file does not stop loading.
    """

    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    root = root.expanduser().resolve()
    log_path = _configure_logging()
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {root}")

    documents: list[Document] = []
    skipped = 0
    failed = 0
    logger.info("Starting LangChain load: data_dir=%s strict=%s", root, strict)

    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        factory = _LOADER_FACTORIES.get(path.suffix.lower())
        if factory is None:
            skipped += 1
            logger.debug("Skipping unsupported file: %s", path)
            continue

        try:
            langchain_loader = factory(path)
            loaded = langchain_loader.load()
            loaded = [
                document
                for document in loaded
                if isinstance(document, Document) and document.page_content.strip()
            ]
        except (ImportError, OSError, UnicodeError, ValueError, RuntimeError) as error:
            if strict:
                raise
            failed += 1
            logger.error("Could not load %s with LangChain: %s", path, error)
            continue

        if not loaded:
            skipped += 1
            logger.warning("Skipping empty document: %s", path)
            continue

        loaded = _add_metadata(loaded, path, root)
        documents.extend(loaded)
        logger.info(
            "Loaded %s with %s (%d LangChain document(s), %d characters)",
            path,
            type(langchain_loader).__name__,
            len(loaded),
            sum(len(document.page_content) for document in loaded),
        )

    logger.info(
        "Completed LangChain load: loaded=%d skipped=%d failed=%d log_file=%s",
        len(documents),
        skipped,
        failed,
        log_path,
    )
    return documents


if __name__ == "__main__":
    loaded_documents = load_documents()
    print(f"Loaded {len(loaded_documents)} LangChain document(s). See logs/loader.log for details.")
