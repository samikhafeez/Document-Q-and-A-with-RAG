"""
main.py — FastAPI application entry point.

Endpoints:
  POST   /api/v1/documents/upload        — upload + ingest a document
  GET    /api/v1/documents               — list all indexed documents
  GET    /api/v1/documents/{doc_id}      — get document info
  DELETE /api/v1/documents/{doc_id}      — remove document from index
  POST   /api/v1/qa/ask                  — ask a question
  GET    /api/v1/qa/history              — recent Q&A history
  GET    /health                         — system status

Start:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import cfg
from app.database import (
    delete_document as db_delete_doc,
    get_document,
    get_qa_history,
    init_db,
    list_documents,
)
from app.models import (
    AnswerResponse,
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QuestionRequest,
)
from app.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.DEBUG if cfg.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Show masked key so you can confirm the right key is loaded without
    # exposing the full secret in logs.
    key = cfg.openai_api_key
    masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
    logger.info("🚀  Starting %s v%s", cfg.app_title, cfg.app_version)
    logger.info("🔑  OpenAI key loaded: %s  model=%s", masked, cfg.chat_model)
    init_db()
    pipeline = RAGPipeline()
    app.state.pipeline = pipeline
    logger.info("✅  RAG pipeline ready")
    yield
    logger.info("👋  Shutdown complete")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=cfg.app_title,
    version=cfg.app_version,
    description="Document Q&A using Retrieval-Augmented Generation (RAG)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


# ── Document endpoints ────────────────────────────────────────────────────────

@app.post(
    "/api/v1/documents/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload and index a document",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    chunking_strategy: str | None = Form(None),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in cfg.supported_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(cfg.supported_extensions)}",
        )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 ** 2)
    if size_mb > cfg.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {cfg.max_file_size_mb} MB.",
        )

    try:
        result = _pipeline(request).ingest_file(file_bytes, file.filename, chunking_strategy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return result


@app.get(
    "/api/v1/documents",
    response_model=DocumentListResponse,
    tags=["Documents"],
    summary="List all indexed documents",
)
async def list_docs():
    rows = list_documents()
    import json
    docs = [
        DocumentInfo(
            doc_id=r["doc_id"],
            filename=r["filename"],
            file_type=r["file_type"],
            num_chunks=r["num_chunks"],
            file_size_bytes=r["file_size_bytes"],
            uploaded_at=r["uploaded_at"],
            metadata=json.loads(r["doc_metadata"] or "{}"),
        )
        for r in rows
    ]
    return DocumentListResponse(documents=docs, total=len(docs))


@app.get(
    "/api/v1/documents/{doc_id}",
    response_model=DocumentInfo,
    tags=["Documents"],
    summary="Get document details",
)
async def get_doc(doc_id: str):
    import json
    row = get_document(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentInfo(
        doc_id=row["doc_id"],
        filename=row["filename"],
        file_type=row["file_type"],
        num_chunks=row["num_chunks"],
        file_size_bytes=row["file_size_bytes"],
        uploaded_at=row["uploaded_at"],
        metadata=json.loads(row["doc_metadata"] or "{}"),
    )


@app.delete(
    "/api/v1/documents/{doc_id}",
    response_model=DeleteResponse,
    tags=["Documents"],
    summary="Remove a document from the index",
)
async def delete_doc(doc_id: str, request: Request):
    removed_vs = _pipeline(request).delete_document(doc_id)
    removed_db = db_delete_doc(doc_id)
    if not removed_vs and not removed_db:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DeleteResponse(doc_id=doc_id, message=f"Document '{doc_id}' removed from index.")


# ── Q&A endpoints ─────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/qa/ask",
    response_model=AnswerResponse,
    tags=["Q&A"],
    summary="Ask a question about your documents",
)
async def ask_question(body: QuestionRequest, request: Request):
    if not _pipeline(request).vector_store.is_loaded:
        raise HTTPException(
            status_code=404,
            detail="No documents are indexed yet. Upload a document first.",
        )
    try:
        return _pipeline(request).answer(
            question=body.question,
            doc_ids=body.doc_ids,
            top_k=body.top_k,
            use_mmr=body.use_mmr,
        )
    except Exception as exc:
        logger.exception("Q&A error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/api/v1/qa/history",
    tags=["Q&A"],
    summary="Recent Q&A history",
)
async def qa_history(limit: int = 20):
    return {"history": get_qa_history(limit=limit)}


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(request: Request):
    vs = _pipeline(request).vector_store
    return HealthResponse(
        status="ok",
        version=cfg.app_version,
        documents_indexed=len(vs.document_ids),
        total_chunks=vs.total_chunks,
        vector_store_loaded=vs.is_loaded,
    )


@app.get("/", tags=["System"])
async def root():
    return JSONResponse({
        "message": f"Welcome to {cfg.app_title}",
        "docs": "/docs",
        "version": cfg.app_version,
    })
