"""
retriever.py — High-level retrieval interface.

Sits between the vector store and the RAG pipeline.
Responsibilities:
  - Execute similarity search (with optional MMR)
  - Hard-filter chunks below a minimum relevance floor
  - Same-page deduplication: keep only the best chunk per (doc_id, page)
  - Token-overlap deduplication for remaining near-duplicate chunks
  - Compute a confidence score that reflects actual retrieval strength
  - Format retrieved chunks into SourceChunk models

Thresholds (all tunable via cfg or constants here):
  _MIN_CHUNK_RELEVANCE  — absolute floor; chunks below this are noise, never shown to LLM
  _SAME_PAGE_KEEP       — max chunks kept per (doc_id, page_number) pair
  _JACCARD_DEDUP        — Jaccard overlap above which two chunks are treated as duplicates
"""

from __future__ import annotations

import logging
from collections import defaultdict

from app.config import cfg
from app.models import SourceChunk
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Absolute minimum cosine similarity — chunks below this score are noise.
# For text-embedding-3-small: >0.50 = clearly relevant, 0.30–0.50 = possibly relevant,
# <0.30 = unlikely to be relevant. Keep at 0.28 so we don't drop marginal edge cases
# at retrieval time; the pipeline can still apply a higher bar before building context.
_MIN_CHUNK_RELEVANCE = 0.28

# Maximum chunks retained from the same document page.
# Prevents "three overlapping chunks from page 2" all reaching the LLM.
_SAME_PAGE_KEEP = 1

# Jaccard token-overlap threshold for near-duplicate detection.
# 0.55 catches same-page overlapping chunks (e.g. chunk_3 and chunk_4 that share
# 60% of words due to chunk_overlap=150). Was 0.97 which caught nothing useful.
_JACCARD_DEDUP = 0.55


class Retriever:
    """
    Wraps VectorStore to provide a clean retrieve() method for the RAG pipeline.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._vs = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        use_mmr: bool | None = None,
    ) -> tuple[list[SourceChunk], float]:
        """
        Retrieve the most relevant chunks for *query*.

        Pipeline:
          1. FAISS search for top_k × 4 candidates (capped at 20)
          2. Hard filter: drop chunks below _MIN_CHUNK_RELEVANCE
          3. Same-page dedup: keep best _SAME_PAGE_KEEP chunk(s) per doc/page
          4. Jaccard dedup: remove overlapping chunks
          5. Return top_k survivors

        Returns:
            (sources, confidence)
            sources    : List of SourceChunk objects sorted by relevance.
            confidence : Score reflecting actual retrieval strength (0–1).
        """
        top_k = top_k if top_k is not None else cfg.top_k

        # Over-fetch — dedup and filtering will reduce the count.
        # Cap at 20 to avoid spending money re-embedding in MMR.
        k_fetch = min(top_k * 4, 20)

        raw_results = self._vs.search(
            query=query,
            top_k=k_fetch,
            doc_ids=doc_ids,
            use_mmr=use_mmr,
        )

        if not raw_results:
            logger.info("No results from vector store for query='%.60s'", query)
            return [], 0.0

        # ── Step 1: hard relevance floor ──────────────────────────────────────
        filtered = [r for r in raw_results if r["relevance_score"] >= _MIN_CHUNK_RELEVANCE]
        dropped  = len(raw_results) - len(filtered)
        if dropped:
            logger.debug(
                "Relevance filter removed %d/%d chunks below %.2f  (top raw score=%.3f)",
                dropped, len(raw_results), _MIN_CHUNK_RELEVANCE,
                raw_results[0]["relevance_score"],
            )

        if not filtered:
            logger.info(
                "All %d candidates below relevance floor %.2f  query='%.60s'",
                len(raw_results), _MIN_CHUNK_RELEVANCE, query,
            )
            return [], 0.0

        # ── Step 2: same-page deduplication ──────────────────────────────────
        page_deduped = self._deduplicate_same_page(filtered)

        # ── Step 3: Jaccard token-overlap deduplication ───────────────────────
        deduped = self._deduplicate_by_overlap(page_deduped)[:top_k]

        # ── Step 4: build SourceChunk objects ────────────────────────────────
        sources = [
            SourceChunk(
                doc_id=r["doc_id"],
                filename=r["filename"],
                chunk_index=r["chunk_index"],
                content=r["content"],
                page_number=r.get("page_number"),
                relevance_score=round(r["relevance_score"], 4),
                section=r.get("section"),
            )
            for r in deduped
        ]

        confidence = self._compute_confidence(deduped)
        logger.info(
            "Retrieved %d chunks  confidence=%.3f  top_score=%.3f  query='%.60s'",
            len(sources), confidence,
            deduped[0]["relevance_score"] if deduped else 0.0,
            query,
        )
        return sources, confidence

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _deduplicate_same_page(self, results: list[dict]) -> list[dict]:
        """
        Within each (doc_id, page_number) group, keep only the top-scoring
        _SAME_PAGE_KEEP chunks.

        This prevents overlapping chunks from the same page all reaching the LLM.
        Chunks with page_number=None are deduplicated per doc_id only.
        """
        # Group by (doc_id, page_number), preserving sort order within groups
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in results:
            key = (r["doc_id"], r.get("page_number"))
            groups[key].append(r)

        survivors: list[dict] = []
        for key, chunks in groups.items():
            # Sort by score descending, keep best _SAME_PAGE_KEEP
            chunks.sort(key=lambda c: c["relevance_score"], reverse=True)
            survivors.extend(chunks[:_SAME_PAGE_KEEP])
            if len(chunks) > _SAME_PAGE_KEEP:
                logger.debug(
                    "Same-page dedup (doc=%s, page=%s): kept %d/%d chunks",
                    key[0][:8], key[1], _SAME_PAGE_KEEP, len(chunks),
                )

        # Re-sort globally by relevance score descending
        survivors.sort(key=lambda c: c["relevance_score"], reverse=True)
        return survivors

    def _deduplicate_by_overlap(self, results: list[dict]) -> list[dict]:
        """
        Remove chunks whose content heavily overlaps with a higher-scoring chunk.

        Uses Jaccard similarity on word tokens. Threshold: _JACCARD_DEDUP (0.55).
        This catches remaining overlapping chunks that survived same-page dedup
        (e.g. identical content appearing in two different pages).
        """
        seen_tokens: list[set] = []
        deduped: list[dict] = []

        for r in results:
            tokens = set(r["content"].lower().split())
            if not tokens:
                continue
            is_dup = any(
                len(tokens & seen) / len(tokens | seen) > _JACCARD_DEDUP
                for seen in seen_tokens
                if seen  # guard against empty sets
            )
            if not is_dup:
                deduped.append(r)
                seen_tokens.append(tokens)

        removed = len(results) - len(deduped)
        if removed:
            logger.debug("Jaccard dedup removed %d overlapping chunks", removed)
        return deduped

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _compute_confidence(self, results: list[dict]) -> float:
        """
        Compute a confidence score that reflects genuine retrieval strength.

        Formula:
          - Primary signal: top chunk's relevance score (weight 0.65)
          - Breadth bonus:  number of chunks with score ≥ 0.45 (up to +0.20)
          - Hard cap:       if top score < 0.35, confidence is capped at 0.35
                            (prevents weak retrieval masquerading as medium confidence)

        For text-embedding-3-small cosine similarities:
          ≥ 0.60  → high confidence (document clearly addresses the question)
          0.40–0.59 → medium confidence (partially addressed)
          < 0.40  → low confidence (should trigger cautious answer or fallback)
        """
        if not results:
            return 0.0

        top_score = results[0]["relevance_score"]

        # Core signal: the top chunk's score is the strongest indicator
        confidence = top_score * 0.65

        # Breadth bonus: more high-quality chunks = more evidence
        strong_chunks = sum(1 for r in results if r["relevance_score"] >= 0.45)
        confidence += min(strong_chunks * 0.06, 0.20)

        # Hard cap for weak top scores — prevents inflated confidence from many
        # mediocre chunks when the best chunk is itself unreliable
        if top_score < 0.35:
            confidence = min(confidence, 0.35)

        return round(min(confidence, 1.0), 4)
