"""
config.py — Central configuration for the Document Q&A RAG system.

All tuneable parameters in one place. Import: from app.config import cfg

Environment variables (from .env or shell) ALWAYS win over the defaults
defined here. load_dotenv() is called here so that config is self-contained
regardless of import order in main.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env FIRST — before any field default is evaluated.
# Safe to call multiple times; subsequent calls are no-ops.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed; env vars must be set externally

ROOT = Path(__file__).resolve().parent.parent

_PLACEHOLDER = "your-openai-api-key"


@dataclass
class Config:
    # ── OpenAI ────────────────────────────────────────────────────────────────
    # Default is empty — the real value MUST come from OPENAI_API_KEY in .env.
    # A placeholder string here will be caught at startup and raise clearly.
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536        # text-embedding-3-small output dim
    temperature: float = 0.1               # low = more factual answers
    max_tokens: int = 1024

    # ── Chunking strategy ─────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 150
    chunking_strategy: str = "recursive"    # "recursive" | "sentence" | "fixed"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 5                          # chunks retrieved per query
    mmr_diversity: float = 0.6             # 0 = max relevance, 1 = max diversity
    use_mmr: bool = True                   # Maximal Marginal Relevance re-ranking
    score_threshold: float = 0.30          # min cosine similarity to include a chunk

    # ── Paths ─────────────────────────────────────────────────────────────────
    upload_dir: Path = ROOT / "data" / "uploads"
    vector_store_dir: Path = ROOT / "data" / "vector_store"
    sample_docs_dir: Path = ROOT / "data" / "sample_docs"
    db_path: Path = ROOT / "data" / "qa_history.db"

    # ── Supported file types ──────────────────────────────────────────────────
    supported_extensions: tuple = (".pdf", ".txt", ".md", ".csv")
    max_file_size_mb: int = 50

    # ── App ───────────────────────────────────────────────────────────────────
    app_title: str = "Document Q&A with RAG"
    app_version: str = "1.0.0"
    debug: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    def __post_init__(self) -> None:
        # ── 1. Override every field from environment variables ────────────────
        # Environment always wins; dataclass defaults are just fallbacks.

        self.openai_api_key    = os.environ.get("OPENAI_API_KEY",     self.openai_api_key)
        self.chat_model        = os.environ.get("CHAT_MODEL",         self.chat_model)
        self.embedding_model   = os.environ.get("EMBEDDING_MODEL",    self.embedding_model)
        self.temperature       = float(os.environ.get("TEMPERATURE",  self.temperature))
        self.max_tokens        = int(os.environ.get("MAX_TOKENS",     self.max_tokens))
        self.chunk_size        = int(os.environ.get("CHUNK_SIZE",     self.chunk_size))
        self.chunk_overlap     = int(os.environ.get("CHUNK_OVERLAP",  self.chunk_overlap))
        self.chunking_strategy = os.environ.get("CHUNKING_STRATEGY",  self.chunking_strategy)
        self.top_k             = int(os.environ.get("TOP_K",            self.top_k))
        self.mmr_diversity     = float(os.environ.get("MMR_DIVERSITY",  self.mmr_diversity))
        self.score_threshold   = float(os.environ.get("SCORE_THRESHOLD", self.score_threshold))
        self.max_file_size_mb  = int(os.environ.get("MAX_FILE_SIZE_MB", self.max_file_size_mb))

        # Boolean env vars: anything other than "false"/"0"/"no" is True
        self.use_mmr = os.environ.get("USE_MMR", str(self.use_mmr)).lower() not in ("false", "0", "no")
        self.debug   = os.environ.get("DEBUG",   str(self.debug)).lower()  not in ("false", "0", "no")

        # CORS origins — comma-separated list in env
        if raw := os.environ.get("CORS_ORIGINS"):
            self.cors_origins = [o.strip() for o in raw.split(",") if o.strip()]

        # Path overrides — resolve relative to project root
        if raw := os.environ.get("UPLOAD_DIR"):
            self.upload_dir = (ROOT / raw) if not Path(raw).is_absolute() else Path(raw)
        if raw := os.environ.get("VECTOR_STORE_DIR"):
            self.vector_store_dir = (ROOT / raw) if not Path(raw).is_absolute() else Path(raw)

        # ── 2. Validate API key ───────────────────────────────────────────────
        if not self.openai_api_key or self.openai_api_key.strip() in ("", _PLACEHOLDER):
            raise ValueError(
                "\n\n"
                "❌  OPENAI_API_KEY is missing or still set to the placeholder value.\n"
                "\n"
                "    Fix:\n"
                "      1. Open the .env file in your project root.\n"
                "      2. Set:  OPENAI_API_KEY=sk-...your-real-key...\n"
                "      3. Get a key from: https://platform.openai.com/api-keys\n"
                "      4. Kill the server and restart it (Ctrl-C, then uvicorn again).\n"
                "\n"
                "    Current value starts with: "
                f"'{self.openai_api_key[:12]}...'\n"
            )

        # ── 3. Ensure storage directories exist ───────────────────────────────
        for d in (self.upload_dir, self.vector_store_dir, self.sample_docs_dir):
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
