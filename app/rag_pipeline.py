"""
rag_pipeline.py — The RAG orchestrator.

Full pipeline per query:
  1. Retrieve top-k relevant chunks via VectorStore (+ MMR re-ranking)
  2. Deduplicate near-identical chunks (Retriever)
  3. Build context string with citation metadata
  4. Call LLM with grounded system prompt
  5. Return structured AnswerResponse with sources + confidence + latency

Ingestion pipeline per document:
  1. Parse file (PDF/TXT/MD/CSV) → raw text + pages
  2. Split into overlapping chunks
  3. Embed with OpenAI text-embedding-3-small
  4. Add to FAISS index
  5. Persist vector store + save to DB
"""

from __future__ import annotations

import logging
import re
import time

from langchain_openai import ChatOpenAI

from app.config import cfg
from app.database import save_document, save_qa
from app.embeddings import EmbeddingService
from app.ingest import process_file, ProcessedDocument
from app.models import AnswerResponse, IngestResponse, SourceChunk
from app.prompts import (
    RAG_PROMPT,
    WEAK_CONTEXT_PROMPT,
    FALLBACK_PROMPT,
    FOLLOWUP_PROMPT,
    build_context_string,
)
from app.retriever import Retriever
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Confidence thresholds ─────────────────────────────────────────────────────
# Retriever confidence is primarily driven by the top chunk's cosine similarity.
# For text-embedding-3-small:
#   ≥ 0.60 → top chunk clearly addresses the question  (full RAG)
#   0.38–0.60 → partial match or indirect relevance     (RAG, but prompt is cautious)
#   0.28–0.38 → weak match, context may be tangential   (WEAK_CONTEXT prompt)
#   < 0.28 → effectively no signal                      (pure fallback)

# Below this → use WEAK_CONTEXT_PROMPT (present but probably tangential)
_WEAK_CONFIDENCE_THRESHOLD = 0.28

# Below this → skip context entirely and use FALLBACK_PROMPT
_LOW_CONFIDENCE_THRESHOLD  = 0.38

# Question type detection — regex for definitional patterns.
# Handles both "What is X?" and the contraction "What's X?"
_DEFINITION_RE = re.compile(
    r"^\s*("
    r"what\s+(is|are|was|were)\b|"   # "What is" / "What are"
    r"what'(?:s|re)\b|"              # "What's" / "What're"
    r"define\b|"
    r"what does .+ mean|"
    r"explain what\b|"
    r"describe what\b|"
    r"who\s+(is|are|was|were)\b|"
    r"what exactly (is|are)\b|"
    r"what do you mean by\b"
    r")",
    re.IGNORECASE,
)


class RAGPipeline:
    """
    Singleton-style orchestrator. Instantiate once at app startup.

    All state (vector store, embeddings, LLM) is encapsulated here.
    The FastAPI app routes delegate to this class.
    """

    def __init__(self) -> None:
        self._emb      = EmbeddingService()
        self._vs       = VectorStore(embedding_service=self._emb)
        self._retriever = Retriever(vector_store=self._vs)
        self._llm      = ChatOpenAI(
            model=cfg.chat_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            openai_api_key=cfg.openai_api_key,
        )
        logger.info(
            "RAGPipeline ready  model=%s  docs=%d  chunks=%d",
            cfg.chat_model,
            len(self._vs.document_ids),
            self._vs.total_chunks,
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        chunking_strategy: str | None = None,
    ) -> IngestResponse:
        """
        Process and index a document.

        Steps: parse → chunk → embed → FAISS index → DB record.
        """
        # 1. Parse + chunk
        doc: ProcessedDocument = process_file(file_bytes, filename, chunking_strategy)

        # 2. Embed + add to vector store
        chunks_added = self._vs.add_documents(doc)

        # 3. Persist metadata to DB
        save_document(
            doc_id=doc.doc_id,
            filename=doc.filename,
            file_type=doc.file_type,
            num_chunks=chunks_added,
            file_size_bytes=doc.file_size_bytes,
            metadata=doc.metadata,
        )

        logger.info("Ingested '%s'  doc_id=%s  chunks=%d", filename, doc.doc_id, chunks_added)

        return IngestResponse(
            doc_id=doc.doc_id,
            filename=filename,
            num_chunks=chunks_added,
            message=f"Successfully ingested '{filename}' into {chunks_added} searchable chunks.",
        )

    # ── Q&A ───────────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        use_mmr: bool | None = None,
    ) -> AnswerResponse:
        """
        Answer *question* using retrieved document context.

        Returns a fully structured AnswerResponse with sources and confidence.
        """
        t_start = time.perf_counter()
        q_type  = self._detect_question_type(question)
        logger.info("Question type: %s  query='%.60s'", q_type, question)

        # 1. Retrieve
        sources, confidence = self._retriever.retrieve(
            query=question,
            top_k=top_k,
            doc_ids=doc_ids,
            use_mmr=use_mmr,
        )

        # 2. Route based on confidence tier
        logger.info(
            "Routing: confidence=%.3f  sources=%d  type=%s  "
            "(thresholds: weak=%.2f, low=%.2f)",
            confidence, len(sources), q_type,
            _WEAK_CONFIDENCE_THRESHOLD, _LOW_CONFIDENCE_THRESHOLD,
        )
        if not sources or confidence < _WEAK_CONFIDENCE_THRESHOLD:
            # No usable signal — pure fallback, no context passed to LLM
            answer_text  = self._fallback_answer(question)
            answer_type  = "fallback"
            sources      = []
            confidence   = 0.0

        elif confidence < _LOW_CONFIDENCE_THRESHOLD:
            # Weak signal — context exists but may be tangential; use cautious prompt
            context_str  = build_context_string(sources)
            answer_text  = self._generate_weak_answer(question, context_str)
            answer_type  = "rag_weak"

        else:
            # Good signal — full RAG answer
            context_str  = build_context_string(sources)
            answer_text  = self._generate_answer(question, context_str)
            answer_type  = "rag"

        latency_ms = (time.perf_counter() - t_start) * 1000

        # 3. Build response
        response = AnswerResponse(
            question=question,
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            answer_type=answer_type,
            latency_ms=round(latency_ms, 2),
            model_used=cfg.chat_model,
        )

        # 4. Log to DB (non-fatal)
        try:
            save_qa(
                question_id=response.question_id,
                question=question,
                answer=answer_text,
                doc_ids=doc_ids or self._vs.document_ids,
                confidence=confidence,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.warning("DB logging failed (non-fatal): %s", exc)

        logger.info(
            "answer() → type=%s  confidence=%.3f  sources=%d  latency=%.0fms",
            answer_type, confidence, len(sources), latency_ms,
        )
        return response

    def suggest_followups(self, question: str, answer: str) -> list[str]:
        """Generate 3 follow-up question suggestions."""
        try:
            chain  = FOLLOWUP_PROMPT | self._llm
            result = chain.invoke({"question": question, "answer": answer})
            lines  = [l.strip() for l in result.content.strip().splitlines() if l.strip()]
            return lines[:3]
        except Exception as exc:
            logger.debug("Follow-up generation failed: %s", exc)
            return []

    # ── Document management ───────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> bool:
        return self._vs.delete_document(doc_id)

    @property
    def vector_store(self) -> VectorStore:
        return self._vs

    # ── Internal ──────────────────────────────────────────────────────────────

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_question_type(question: str) -> str:
        """
        Return 'definition' for simple definitional / factual questions,
        'analytical' for everything else.

        Definitional questions get a shorter, tighter answer style.
        """
        return "definition" if _DEFINITION_RE.search(question.strip()) else "analytical"

    def _generate_answer(self, question: str, context: str) -> str:
        """Call the LLM with the full grounded RAG prompt."""
        try:
            chain  = RAG_PROMPT | self._llm
            result = chain.invoke({"context": context, "question": question})
            return result.content.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return (
                "I encountered an error generating a response. "
                "Please try again or check your OpenAI API key."
            )

    def _generate_weak_answer(self, question: str, context: str) -> str:
        """
        Call the LLM with the cautious weak-context prompt.
        Used when retrieval returned something but confidence is below the
        full-RAG threshold.  The prompt explicitly tells the LLM to be
        conservative and acknowledge partial coverage.
        """
        try:
            chain  = WEAK_CONTEXT_PROMPT | self._llm
            result = chain.invoke({"context": context, "question": question})
            return result.content.strip()
        except Exception as exc:
            logger.error("Weak-context LLM call failed: %s", exc)
            return self._fallback_answer(question)

    def _fallback_answer(self, question: str) -> str:
        """Return a polite no-context answer."""
        try:
            chain  = FALLBACK_PROMPT | self._llm
            result = chain.invoke({"question": question})
            return result.content.strip()
        except Exception:
            return (
                "No relevant information was found in the uploaded documents "
                "for your question. Please try uploading relevant documents "
                "or rephrasing your question."
            )
