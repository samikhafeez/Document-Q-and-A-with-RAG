"""
ingest.py — Document ingestion and chunking pipeline.

Supported formats : PDF, TXT, Markdown, CSV
Chunking strategies:
  recursive  — RecursiveCharacterTextSplitter (default, best for mixed content)
  sentence   — splits on sentence boundaries (best for Q&A over prose)
  fixed      — fixed token/char size (fast baseline)

Each chunk carries rich metadata:
  doc_id, filename, chunk_index, total_chunks,
  page_number (PDF only), char_start, char_end, section (heading detected)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_core.documents import Document

from app.config import cfg

logger = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ProcessedDocument:
    doc_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunks: list[Document]              # LangChain Documents with metadata
    raw_text: str = ""
    page_count: int = 0
    metadata: dict = field(default_factory=dict)


# ── Document ID ───────────────────────────────────────────────────────────────

def make_doc_id(filename: str, content: bytes) -> str:
    """Deterministic ID: SHA-256 of filename + first 4 KB of content."""
    h = hashlib.sha256(filename.encode() + content[:4096]).hexdigest()
    return h[:16]


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> tuple[str, int]:
    """
    Extract text from PDF bytes.  Uses PyMuPDF (fitz) as primary,
    falls back to pypdf if fitz is unavailable.

    Returns (full_text, page_count).
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages), len(doc)
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
        from io import BytesIO
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages), len(reader.pages)
    except ImportError:
        raise ImportError(
            "PDF extraction requires either PyMuPDF or pypdf.\n"
            "Install with: pip install pymupdf  OR  pip install pypdf"
        )


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode plain-text file (UTF-8 with replacement for bad bytes)."""
    return file_bytes.decode("utf-8", errors="replace")


def extract_text_from_csv(file_bytes: bytes) -> str:
    """Convert CSV to readable text (header + rows)."""
    import csv
    from io import StringIO
    text = file_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(StringIO(text))
    lines = []
    for row in reader:
        lines.append("  |  ".join(f"{k}: {v}" for k, v in row.items()))
    return "\n".join(lines)


# ── Section detector ──────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,4}\s+.+|[A-Z][A-Z\s]{4,50}:?\s*)$", re.MULTILINE)

def detect_section(text: str) -> str | None:
    """Return the last Markdown heading or ALL-CAPS section title found in text."""
    matches = _HEADING_RE.findall(text)
    return matches[-1].strip() if matches else None


# ── Chunking strategies ───────────────────────────────────────────────────────

def _get_splitter(strategy: str | None = None):
    strategy = strategy or cfg.chunking_strategy
    if strategy == "sentence":
        # Sentence-aware splitter: splits on '.', '?', '!', '\n\n'
        return RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=[". ", "? ", "! ", "\n\n", "\n", " ", ""],
            length_function=len,
        )
    elif strategy == "fixed":
        return CharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separator="\n",
        )
    else:  # "recursive" (default)
        return RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )


# ── Page-aware chunking for PDFs ──────────────────────────────────────────────

def _chunk_pdf_text(
    full_text: str,
    doc_id: str,
    filename: str,
    page_count: int,
    strategy: str | None = None,
) -> list[Document]:
    """
    Split PDF text into chunks while preserving page number metadata.
    Sections are extracted from [Page N] markers in the text.
    """
    splitter = _get_splitter(strategy)
    # Split by page markers first to track page numbers
    page_pattern = re.compile(r"\[Page (\d+)\]\n")
    parts = page_pattern.split(full_text)

    # parts = ["", "1", "<page1 text>", "2", "<page2 text>", ...]
    chunks: list[Document] = []
    char_cursor = 0
    i = 1  # skip leading empty string

    while i < len(parts) - 1:
        page_num = int(parts[i])
        page_text = parts[i + 1]
        page_chunks = splitter.split_text(page_text)
        for j, chunk_text in enumerate(page_chunks):
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "doc_id":      doc_id,
                    "filename":    filename,
                    "chunk_index": len(chunks),
                    "page_number": page_num,
                    "section":     detect_section(chunk_text),
                    "char_start":  char_cursor,
                    "char_end":    char_cursor + len(chunk_text),
                },
            ))
            char_cursor += len(chunk_text)
        i += 2

    # Re-index
    for idx, doc in enumerate(chunks):
        doc.metadata["chunk_index"] = idx
        doc.metadata["total_chunks"] = len(chunks)

    return chunks


def _chunk_plain_text(
    text: str,
    doc_id: str,
    filename: str,
    strategy: str | None = None,
) -> list[Document]:
    """Chunk plain text with positional and section metadata."""
    splitter = _get_splitter(strategy)
    raw_chunks = splitter.split_text(text)
    char_cursor = 0
    docs = []
    for i, chunk_text in enumerate(raw_chunks):
        docs.append(Document(
            page_content=chunk_text,
            metadata={
                "doc_id":      doc_id,
                "filename":    filename,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "page_number": None,
                "section":     detect_section(chunk_text),
                "char_start":  char_cursor,
                "char_end":    char_cursor + len(chunk_text),
            },
        ))
        char_cursor += len(chunk_text)
    return docs


# ── Main processor ────────────────────────────────────────────────────────────

def process_file(
    file_bytes: bytes,
    filename: str,
    chunking_strategy: str | None = None,
) -> ProcessedDocument:
    """
    Full ingestion pipeline for a single file.

    1. Detect file type from extension.
    2. Extract raw text (with page awareness for PDFs).
    3. Split into chunks with metadata.
    4. Return ProcessedDocument ready for embedding.
    """
    ext = Path(filename).suffix.lower()
    if ext not in cfg.supported_extensions:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(cfg.supported_extensions)}"
        )

    doc_id = make_doc_id(filename, file_bytes)
    logger.info("Processing '%s'  type=%s  size=%.1f KB", filename, ext, len(file_bytes) / 1024)

    page_count = 0

    if ext == ".pdf":
        raw_text, page_count = extract_text_from_pdf(file_bytes, filename)
        chunks = _chunk_pdf_text(raw_text, doc_id, filename, page_count, chunking_strategy)
    elif ext in (".txt", ".md"):
        raw_text = extract_text_from_txt(file_bytes)
        chunks = _chunk_plain_text(raw_text, doc_id, filename, chunking_strategy)
    elif ext == ".csv":
        raw_text = extract_text_from_csv(file_bytes)
        chunks = _chunk_plain_text(raw_text, doc_id, filename, chunking_strategy)
    else:
        raw_text = extract_text_from_txt(file_bytes)
        chunks = _chunk_plain_text(raw_text, doc_id, filename, chunking_strategy)

    if not chunks:
        raise ValueError(f"No text could be extracted from '{filename}'.")

    logger.info(
        "  ✓ '%s' → %d chunks  (pages=%d, avg_chunk_len=%d chars)",
        filename, len(chunks), page_count,
        sum(len(c.page_content) for c in chunks) // max(len(chunks), 1),
    )

    return ProcessedDocument(
        doc_id=doc_id,
        filename=filename,
        file_type=ext.lstrip("."),
        file_size_bytes=len(file_bytes),
        chunks=chunks,
        raw_text=raw_text,
        page_count=page_count,
        metadata={
            "page_count":       page_count,
            "total_chars":      len(raw_text),
            "chunking_strategy": chunking_strategy or cfg.chunking_strategy,
            "chunk_size":       cfg.chunk_size,
            "chunk_overlap":    cfg.chunk_overlap,
        },
    )
