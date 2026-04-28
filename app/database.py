"""
database.py — SQLite persistence layer.

Tables:
  documents   — one row per ingested document (metadata + chunk count)
  qa_history  — one row per question asked

Uses SQLAlchemy Core (no ORM) for minimal overhead.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, DateTime, Integer, JSON, MetaData,
    String, Table, Text, create_engine, select, delete, update,
)
from sqlalchemy.dialects.sqlite import insert   # gives us on_conflict_do_update

from app.config import cfg

logger = logging.getLogger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────

cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
_engine = create_engine(
    f"sqlite:///{cfg.db_path}",
    connect_args={"check_same_thread": False},
    echo=cfg.debug,
)
_meta = MetaData()

# ── Table definitions ─────────────────────────────────────────────────────────

documents_table = Table(
    "documents", _meta,
    Column("doc_id",           String(64),  primary_key=True),
    Column("filename",         String(256), nullable=False),
    Column("file_type",        String(16),  nullable=False),
    Column("num_chunks",       Integer,     nullable=False, default=0),
    Column("file_size_bytes",  Integer,     nullable=False, default=0),
    Column("uploaded_at",      DateTime,    default=datetime.utcnow),
    Column("doc_metadata",     Text,        nullable=True),   # JSON string
)

qa_history_table = Table(
    "qa_history", _meta,
    Column("question_id",  String(64),  primary_key=True),
    Column("question",     Text,        nullable=False),
    Column("answer",       Text,        nullable=False),
    Column("doc_ids",      Text,        nullable=True),   # JSON list
    Column("confidence",   String(16),  nullable=True),
    Column("latency_ms",   String(16),  nullable=True),
    Column("created_at",   DateTime,    default=datetime.utcnow),
)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    _meta.create_all(_engine)
    logger.info("Database tables ready at %s", cfg.db_path)


# ── Document operations ───────────────────────────────────────────────────────

def save_document(
    doc_id: str,
    filename: str,
    file_type: str,
    num_chunks: int,
    file_size_bytes: int,
    metadata: dict | None = None,
) -> None:
    """Insert or update a document record (upsert on doc_id)."""
    values = dict(
        doc_id=doc_id,
        filename=filename,
        file_type=file_type,
        num_chunks=num_chunks,
        file_size_bytes=file_size_bytes,
        uploaded_at=datetime.utcnow(),
        doc_metadata=json.dumps(metadata or {}),
    )
    stmt = insert(documents_table).values(**values)
    # On duplicate doc_id, update all mutable columns so re-uploads work cleanly
    stmt = stmt.on_conflict_do_update(
        index_elements=["doc_id"],
        set_={k: v for k, v in values.items() if k != "doc_id"},
    )
    with _engine.begin() as conn:
        conn.execute(stmt)


def get_document(doc_id: str) -> dict | None:
    with _engine.connect() as conn:
        row = conn.execute(
            select(documents_table).where(documents_table.c.doc_id == doc_id)
        ).fetchone()
    return row._asdict() if row else None


def list_documents() -> list[dict]:
    with _engine.connect() as conn:
        rows = conn.execute(
            select(documents_table).order_by(documents_table.c.uploaded_at.desc())
        ).fetchall()
    return [r._asdict() for r in rows]


def delete_document(doc_id: str) -> bool:
    with _engine.begin() as conn:
        result = conn.execute(
            delete(documents_table).where(documents_table.c.doc_id == doc_id)
        )
    return result.rowcount > 0


def update_chunk_count(doc_id: str, num_chunks: int) -> None:
    with _engine.begin() as conn:
        conn.execute(
            update(documents_table)
            .where(documents_table.c.doc_id == doc_id)
            .values(num_chunks=num_chunks)
        )


# ── Q&A history ───────────────────────────────────────────────────────────────

def save_qa(
    question_id: str,
    question: str,
    answer: str,
    doc_ids: list[str],
    confidence: float,
    latency_ms: float,
) -> None:
    with _engine.begin() as conn:
        conn.execute(insert(qa_history_table).values(
            question_id=question_id,
            question=question,
            answer=answer,
            doc_ids=json.dumps(doc_ids),
            confidence=str(round(confidence, 4)),
            latency_ms=str(round(latency_ms, 2)),
            created_at=datetime.utcnow(),
        ))


def get_qa_history(limit: int = 50) -> list[dict]:
    with _engine.connect() as conn:
        rows = conn.execute(
            select(qa_history_table)
            .order_by(qa_history_table.c.created_at.desc())
            .limit(limit)
        ).fetchall()
    return [r._asdict() for r in rows]
