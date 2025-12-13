from __future__ import annotations

import logging
import os
from http import HTTPStatus

import rag_store
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from rag_store import RAGChunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])

_session = requests.Session()


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# -------------------------------------------------------------------
# Request / response models
# -------------------------------------------------------------------


class IngestRequest(BaseModel):
    documents: list[str] = Field(..., description="Raw texts to ingest into the vector store.")


class IngestResponse(BaseModel):
    ingested: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of similar chunks to retrieve from the vector store.",
    )


class ChunkOut(BaseModel):
    id: str | None = None
    text: str
    distance: float | None = None
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    context: list[str]
    chunks: list[ChunkOut] = Field(default_factory=list)


class StatusResponse(BaseModel):
    docs_dir: str
    json_files: int
    chunks_in_store: int
    files: list[str] = Field(default_factory=list)


class ReindexResponse(BaseModel):
    documents: int
    chunks: int
    files: int


# -------------------------------------------------------------------
# Ollama chat wrapper
# -------------------------------------------------------------------


def _get_ollama_chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")


def _get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")


def _call_ollama_chat(*, question: str, context: str) -> str:
    """Call Ollama's /api/chat endpoint with a simple RAG-style prompt."""
    base_url = _get_ollama_base_url()
    model = _get_ollama_chat_model()

    prompt = (
        "Use the context below aside from general knowledge to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You answer using the given context."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    # Use the module-level session so tests can monkeypatch `_session`.
    resp = _session.post(f"{base_url}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    message = data.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama chat response missing 'message.content'")

    return content


def _chunk_id_from_metadata(meta: dict) -> str | None:
    doc_id = meta.get("doc_id")
    idx = meta.get("chunk_index")
    if idx is None:
        idx = meta.get("index")
    if isinstance(doc_id, str) and doc_id and idx is not None:
        return f"{doc_id}:{idx}"
    return None


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
def ingest_rag(request: IngestRequest) -> IngestResponse:
    """(Optional) Ingest raw texts (kept for backwards-compat / testing).

    In production, prefer indexing from JSON files via /rag/reindex or startup auto-index.
    """
    docs = [d.strip() for d in request.documents if d and d.strip()]

    if not docs:
        # test_rag_ingest_empty_documents_returns_400
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No documents provided.",
        )

    successes = 0
    last_error: Exception | None = None

    for text in docs:
        try:
            rag_store.add_document(text)
            successes += 1
        except Exception as exc:
            logger.exception("Failed to ingest document", exc_info=exc)
            last_error = exc

    if successes == 0 and last_error is not None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f"Document ingestion failed: {last_error}",
        )

    return IngestResponse(ingested=successes)


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    docs_dir = os.getenv("DOCS_DIR", "/data/docs")
    file_paths = rag_store.list_json_files(docs_dir)
    file_names = [os.path.basename(p) for p in file_paths][:50]

    return StatusResponse(
        docs_dir=docs_dir,
        json_files=len(file_paths),
        chunks_in_store=rag_store.get_collection_count(),
        files=file_names,
    )


@router.post("/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    """Clear and rebuild the vector DB from JSON files in DOCS_DIR."""
    docs_dir = os.getenv("DOCS_DIR", "/data/docs")
    enabled = _truthy(os.getenv("RAG_REINDEX_ENABLED", "true"))
    if not enabled:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Reindex is disabled by configuration.",
        )

    try:
        stats = rag_store.rebuild_from_json_dir(docs_dir)
        return ReindexResponse(**stats)
    except Exception as exc:
        logger.exception("Reindex failed", exc_info=exc)
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    """Run a full RAG cycle: retrieve similar chunks and ask the chat model."""
    try:
        chunks: list[RAGChunk] = rag_store.query_similar_chunks(
            request.question,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.exception("Vector search failed", exc_info=exc)
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f"Vector search failed: {exc}",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="No relevant context found for the given question.",
        )

    context_texts = [c.text for c in chunks]
    context_block = "\n\n".join(context_texts)

    try:
        answer = _call_ollama_chat(
            question=request.question,
            context=context_block,
        )
    except Exception as exc:
        logger.exception("Ollama chat failed", exc_info=exc)
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    chunk_out: list[ChunkOut] = []
    for c in chunks:
        meta = c.metadata if isinstance(c.metadata, dict) else {}
        chunk_out.append(
            ChunkOut(
                id=_chunk_id_from_metadata(meta),
                text=c.text,
                distance=c.distance,
                metadata=meta,
            )
        )

    return QueryResponse(answer=answer, context=context_texts, chunks=chunk_out)
