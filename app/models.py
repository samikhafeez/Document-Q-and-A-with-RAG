"""
models.py — Pydantic schemas for API requests and responses.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Document management ───────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    num_chunks: int
    file_size_bytes: int
    uploaded_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    message: str
    status: str = "success"


class DeleteResponse(BaseModel):
    doc_id: str
    message: str
    status: str = "success"


# ── Q&A ───────────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    doc_ids: list[str] | None = Field(
        None,
        description="Specific document IDs to query. None = query all documents.",
    )
    top_k: int | None = Field(None, ge=1, le=20)
    use_mmr: bool | None = None


class SourceChunk(BaseModel):
    """A retrieved document chunk used to generate the answer."""
    doc_id: str
    filename: str
    chunk_index: int
    content: str
    page_number: int | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    section: str | None = None        # heading / section name if detected


class AnswerResponse(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    sources: list[SourceChunk]
    confidence: float = Field(ge=0.0, le=1.0, description="Mean relevance of top sources")
    answer_type: str = "rag"          # "rag" | "fallback" | "no_context"
    latency_ms: float
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    documents_indexed: int
    total_chunks: int
    vector_store_loaded: bool
