"""
embeddings.py — OpenAI embedding wrapper with batching and caching.

Why batch?  OpenAI's embedding API accepts up to 2048 inputs per call.
Batching reduces API calls and latency when ingesting large documents.

Caching: an in-memory LRU cache avoids re-embedding identical text
(e.g. repeated queries or duplicate chunks across documents).
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import numpy as np
from langchain_openai import OpenAIEmbeddings

from app.config import cfg

logger = logging.getLogger(__name__)

_BATCH_SIZE = 512          # OpenAI recommended batch size for throughput
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 1.0         # seconds


class EmbeddingService:
    """
    Thin wrapper around LangChain's OpenAIEmbeddings.

    Adds:
      - Configurable batch size to avoid API payload limits
      - Exponential-backoff retry on rate-limit errors
      - Per-text LRU cache (saves cost on repeated queries)
      - Helper to return raw numpy arrays for FAISS
    """

    def __init__(self) -> None:
        self._embedder = OpenAIEmbeddings(
            model=cfg.embedding_model,
            openai_api_key=cfg.openai_api_key,
            dimensions=cfg.embedding_dimensions,
        )
        logger.info("EmbeddingService ready  model=%s  dim=%d",
                    cfg.embedding_model, cfg.embedding_dimensions)

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts in batches.

        Returns a list of embedding vectors (one per text).
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE

        for batch_idx in range(total_batches):
            batch = texts[batch_idx * _BATCH_SIZE : (batch_idx + 1) * _BATCH_SIZE]
            embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)

            if total_batches > 1:
                logger.debug(
                    "Embedded batch %d/%d  (%d texts)",
                    batch_idx + 1, total_batches, len(batch),
                )

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Uses an LRU cache so repeated identical queries hit no API.
        """
        return self._cached_embed_query(text)

    def embed_texts_as_array(self, texts: list[str]) -> np.ndarray:
        """Return embeddings as a (N, dim) float32 numpy array."""
        vecs = self.embed_texts(texts)
        return np.array(vecs, dtype=np.float32)

    def get_langchain_embedder(self) -> OpenAIEmbeddings:
        """Return the underlying LangChain embedder (used by FAISS constructor)."""
        return self._embedder

    # ── Internal ──────────────────────────────────────────────────────────────

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embedding API with exponential-backoff retry."""
        import openai
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                return self._embedder.embed_documents(texts)
            except openai.RateLimitError:
                wait = _RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("Rate limit hit; retrying in %.1fs (%d/%d)", wait, attempt, _RETRY_ATTEMPTS)
                if attempt == _RETRY_ATTEMPTS:
                    raise
                time.sleep(wait)
            except Exception:
                raise

    @lru_cache(maxsize=512)
    def _cached_embed_query(self, text: str) -> tuple:
        """LRU-cached single-text embedding. Returns a tuple (hashable for cache)."""
        vec = self._embedder.embed_query(text)
        return tuple(vec)

    def embed_query(self, text: str) -> list[float]:
        return list(self._cached_embed_query(text))
