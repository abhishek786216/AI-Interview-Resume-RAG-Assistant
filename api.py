"""HTTP API for the LangChain RAG services."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.interview.evaluator import evaluate_answer
from src.interview.questions import generate_interview_question
from src.rag.qa import answer_question
from src.retrieval.retriever import retrieve_documents
from src.retrieval.vectorstore import build_and_save_vector_store

app = FastAPI(title="AI Interview & Resume RAG API")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
ALLOWED_CATEGORIES = {"resume", "projects", "job_descriptions"}
REPLACE_ON_UPLOAD = {"resume", "job_descriptions"}
REBUILD_LOCK = threading.Lock()
UPLOAD_SESSION_LOCK = threading.Lock()
UPLOAD_SESSION_STARTED = False


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=10)


class InterviewRequest(BaseModel):
    focus: str = Field(default="candidate projects and technical decisions", min_length=1)
    k: int = Field(default=5, ge=1, le=10)


class EvaluationRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    source_paths: list[str] = Field(min_length=1)


def _sources(documents) -> list[dict[str, object]]:
    return [
        {
            "source": document.metadata.get("relative_source", "unknown"),
            "chunk": document.metadata.get("chunk_index", "unknown"),
            "content": document.page_content,
        }
        for document in documents
    ]


def _replace_previous_documents(category: str, target: Path) -> None:
    """Keep one active resume and job description for the web upload flow."""

    if category not in REPLACE_ON_UPLOAD:
        return
    for existing in target.iterdir():
        if existing.is_file() and existing.name != ".gitkeep":
            existing.unlink()


def _start_upload_session() -> None:
    """Remove bundled documents before accepting the first upload in this run."""

    global UPLOAD_SESSION_STARTED
    if UPLOAD_SESSION_STARTED:
        return

    with UPLOAD_SESSION_LOCK:
        if UPLOAD_SESSION_STARTED:
            return
        for category in ALLOWED_CATEGORIES:
            target = DATA_ROOT / category
            target.mkdir(parents=True, exist_ok=True)
            for existing in target.iterdir():
                if existing.is_file() and existing.name != ".gitkeep":
                    existing.unlink()
        UPLOAD_SESSION_STARTED = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload/{category}")
async def upload_document(
    category: Literal["resume", "projects", "job_descriptions"],
    file: UploadFile = File(...),
) -> dict[str, str]:
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(status_code=400, detail="Unsupported upload category")
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="A file name is required")

    _start_upload_session()
    target_dir = DATA_ROOT / category
    target_dir.mkdir(parents=True, exist_ok=True)
    _replace_previous_documents(category, target_dir)
    target = target_dir / filename
    target.write_bytes(await file.read())
    return {"category": category, "filename": filename}


@app.post("/rebuild")
def rebuild() -> dict[str, int]:
    try:
        with REBUILD_LOCK:
            store = build_and_save_vector_store()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {"vectors": store.index.ntotal}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    try:
        answer, documents = answer_question(request.question, k=request.k)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"answer": answer, "sources": _sources(documents)}


@app.post("/interview/question")
def interview_question(request: InterviewRequest) -> dict[str, object]:
    try:
        result = generate_interview_question(request.focus, k=request.k)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"question": result.question, "sources": _sources(result.sources)}


@app.post("/interview/evaluate")
def interview_evaluate(request: EvaluationRequest) -> dict[str, object]:
    try:
        documents = retrieve_documents(request.question, k=10)
        selected = [
            document
            for document in documents
            if document.metadata.get("relative_source") in request.source_paths
        ] or documents
        result = evaluate_answer(request.question, request.answer, selected)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "score": result.score,
        "correct_points": result.correct_points,
        "missing_points": result.missing_points,
        "feedback": result.feedback,
    }
