"""
vector_store.py — FAISS vector store with document-level management.

Architecture decisions:
  - One unified FAISS index for all documents (simpler, faster at small scale)
  - Chunk metadata stored separately in a dict (FAISS only stores vectors)
  - Cosine similarity via normalised inner-product (IndexFlatIP)
  - Per-document deletion implemented via metadata filtering + index rebuild
  - Persisted to disk after every write operation

FAISS index type: IndexFlatIP (exact search, cosine similarity)
  - No approximation error → correct top-k every time
  - Suitable for up to ~500k chunks; switch to IndexIVFFlat beyond that
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document

from app.config import cfg
from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

_INDEX_FILE  = cfg.vector_store_dir / "faiss.index"
_META_FILE   = cfg.vector_store_dir / "metadata.pkl"
_DOCMAP_FILE = cfg.vector_store_dir / "doc_map.json"


class VectorStore:
    """
    Manages a FAISS index + associated chunk metadata.

    Public interface:
        add_documents(doc: ProcessedDocument)   — embed + add chunks to index
        search(query, top_k, doc_ids)           — return top-k relevant chunks
        delete_document(doc_id)                 — remove all chunks for a doc
        persist() / load()                      — disk I/O
    """

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self._emb   = embedding_service or EmbeddingService()
        self._dim   = cfg.embedding_dimensions
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[dict] = []          # parallel to index rows
        self._doc_map: dict[str, list[int]] = {}  # doc_id → list of chunk indices

        self._load_if_exists()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def document_ids(self) -> list[str]:
        return list(self._doc_map.keys())

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_documents(self, processed_doc) -> int:
        """
        Embed all chunks in *processed_doc* and add to the FAISS index.

        Returns the number of chunks added.
        """
        from app.ingest import ProcessedDocument
        chunks: list[Document] = processed_doc.chunks
        doc_id: str = processed_doc.doc_id

        if not chunks:
            logger.warning("add_documents(): no chunks for doc_id=%s", doc_id)
            return 0

        # If document already exists, remove old version first
        if doc_id in self._doc_map:
            logger.info("Re-ingesting doc_id=%s (removing old chunks)", doc_id)
            self._remove_doc_chunks(doc_id)

        texts = [c.page_content for c in chunks]
        logger.info("Embedding %d chunks for '%s' …", len(texts), processed_doc.filename)
        vectors = self._emb.embed_texts_as_array(texts)
        vectors = self._normalise(vectors)

        start_idx = len(self._chunks)
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)

        self._index.add(vectors)

        # Store metadata for each chunk
        new_indices = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            global_idx = start_idx + i
            self._chunks.append({
                "global_idx":  global_idx,
                "doc_id":      doc_id,
                "filename":    processed_doc.filename,
                "chunk_index": chunk.metadata.get("chunk_index", i),
                "total_chunks": chunk.metadata.get("total_chunks", len(chunks)),
                "page_number": chunk.metadata.get("page_number"),
                "section":     chunk.metadata.get("section"),
                "char_start":  chunk.metadata.get("char_start", 0),
                "char_end":    chunk.metadata.get("char_end", 0),
                "content":     chunk.page_content,
            })
            new_indices.append(global_idx)

        self._doc_map[doc_id] = new_indices
        logger.info("Added %d chunks for doc_id=%s  (total=%d)", len(chunks), doc_id, len(self._chunks))
        self.persist()
        return len(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        use_mmr: bool | None = None,
    ) -> list[dict]:
        """
        Find the most relevant chunks for *query*.

        Args:
            query   : User question string
            top_k   : Number of chunks to return
            doc_ids : Restrict search to specific documents (None = all)
            use_mmr : Use Maximal Marginal Relevance re-ranking

        Returns:
            List of chunk dicts sorted by relevance_score descending.
            Each dict has: content, doc_id, filename, page_number,
                           chunk_index, section, relevance_score.
        """
        if not self.is_loaded:
            return []

        top_k   = top_k   if top_k   is not None else cfg.top_k
        use_mmr = use_mmr if use_mmr is not None else cfg.use_mmr

        # Embed query
        q_vec = np.array([self._emb.embed_query(query)], dtype=np.float32)
        q_vec = self._normalise(q_vec)

        # Retrieve more candidates when using MMR (we'll filter down)
        k_candidates = min(top_k * 3 if use_mmr else top_k, len(self._chunks))

        scores, indices = self._index.search(q_vec, k_candidates)

        # Build candidate list
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if doc_ids and chunk["doc_id"] not in doc_ids:
                continue
            if score < cfg.score_threshold:
                continue
            candidates.append({**chunk, "relevance_score": float(np.clip(score, 0.0, 1.0))})

        if not candidates:
            return []

        # MMR re-ranking for diversity
        if use_mmr and len(candidates) > top_k:
            candidates = self._mmr_rerank(candidates, q_vec[0], top_k)
        else:
            candidates = candidates[:top_k]

        return sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)

    # ── Deletion ──────────────────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks for *doc_id* and rebuild the FAISS index.

        FAISS IndexFlatIP does not support in-place deletion, so we rebuild
        by re-embedding only the remaining chunks (fast because we cache vectors).
        """
        if doc_id not in self._doc_map:
            return False

        self._remove_doc_chunks(doc_id)
        self.persist()
        logger.info("Deleted doc_id=%s from vector store", doc_id)
        return True

    def _remove_doc_chunks(self, doc_id: str) -> None:
        """Internal: remove chunks and rebuild index without persisting."""
        old_indices = set(self._doc_map.pop(doc_id, []))
        self._chunks = [c for c in self._chunks if c["global_idx"] not in old_indices]

        # Re-index global_idx values
        for i, chunk in enumerate(self._chunks):
            chunk["global_idx"] = i

        # Rebuild doc_map with new indices
        new_doc_map: dict[str, list[int]] = {}
        for chunk in self._chunks:
            new_doc_map.setdefault(chunk["doc_id"], []).append(chunk["global_idx"])
        self._doc_map = new_doc_map

        # Rebuild FAISS index (re-embed from stored content)
        if self._chunks:
            texts = [c["content"] for c in self._chunks]
            vecs = self._emb.embed_texts_as_array(texts)
            vecs = self._normalise(vecs)
            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(vecs)
        else:
            self._index = None

    # ── MMR ───────────────────────────────────────────────────────────────────

    def _mmr_rerank(
        self,
        candidates: list[dict],
        query_vec: np.ndarray,
        top_k: int,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance re-ranking.

        Iteratively selects chunks that are relevant to the query AND
        diverse from already-selected chunks. Controlled by cfg.mmr_diversity.
        λ=0 → pure relevance, λ=1 → pure diversity.
        """
        λ = cfg.mmr_diversity
        selected: list[dict] = []
        remaining = list(candidates)

        # Pre-embed candidate content for inter-chunk similarity.
        # Convert to a plain Python list of 1-D row vectors immediately so
        # that .pop(best_idx) works correctly throughout the loop.
        cand_texts = [c["content"] for c in remaining]
        cand_vecs: list[np.ndarray] = list(
            self._normalise(self._emb.embed_texts_as_array(cand_texts))
        )

        selected_vecs: list[np.ndarray] = []

        while remaining and len(selected) < top_k:
            if not selected_vecs:
                # First pick: highest relevance
                best_idx = max(range(len(remaining)), key=lambda i: remaining[i]["relevance_score"])
            else:
                # MMR score = λ * relevance - (1-λ) * max_similarity_to_selected
                scores = []
                for i, chunk in enumerate(remaining):
                    rel = chunk["relevance_score"]
                    sim_to_selected = max(
                        float(np.dot(cand_vecs[i], sv))
                        for sv in selected_vecs
                    )
                    mmr_score = λ * rel - (1 - λ) * sim_to_selected
                    scores.append(mmr_score)
                best_idx = scores.index(max(scores))

            selected.append(remaining.pop(best_idx))
            selected_vecs.append(cand_vecs.pop(best_idx))

        return selected

    # ── Persistence ───────────────────────────────────────────────────────────

    def persist(self) -> None:
        """Save FAISS index, chunk metadata, and doc map to disk."""
        if self._index is None:
            return
        cfg.vector_store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(_INDEX_FILE))
        with open(_META_FILE, "wb") as f:
            pickle.dump(self._chunks, f)
        with open(_DOCMAP_FILE, "w") as f:
            json.dump(self._doc_map, f)
        logger.info("Vector store persisted (%d chunks, %d docs)", len(self._chunks), len(self._doc_map))

    def _load_if_exists(self) -> None:
        if _INDEX_FILE.exists() and _META_FILE.exists() and _DOCMAP_FILE.exists():
            try:
                self._index = faiss.read_index(str(_INDEX_FILE))
                with open(_META_FILE, "rb") as f:
                    self._chunks = pickle.load(f)
                with open(_DOCMAP_FILE) as f:
                    self._doc_map = json.load(f)
                logger.info(
                    "Loaded vector store: %d chunks across %d documents",
                    len(self._chunks), len(self._doc_map),
                )
            except Exception as exc:
                logger.warning("Could not load vector store: %s — starting fresh", exc)
                self._index, self._chunks, self._doc_map = None, [], {}
        else:
            logger.info("No existing vector store — will build on first ingest")

    @staticmethod
    def _normalise(mat: np.ndarray) -> np.ndarray:
        """L2-normalise rows so inner-product equals cosine similarity."""
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return mat / norms
