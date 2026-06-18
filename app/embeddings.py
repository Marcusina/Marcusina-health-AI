"""
Shared sentence-embedding singleton.

One self-hosted sentence-transformer (CPU, no GPU, no API) reused by every
embedding consumer — the RAG retriever and the content search/recommend store —
so the model is loaded into memory exactly once per process.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
from loguru import logger

from app.core.config import get_settings

settings = get_settings()


class Embedder(Protocol):
    def encode(self, texts: list[str], **kwargs) -> np.ndarray: ...


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[embeddings] loading {settings.HF_EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(
            settings.HF_EMBEDDING_MODEL,
            cache_folder=os.path.join(settings.MODELS_DIR, "embedders"),
        )
    return _embedder


def embed(texts: list[str], embedder: Embedder | None = None) -> np.ndarray:
    """Encode texts to a float32 matrix (n, dim). Not normalized — caller decides."""
    e = embedder or get_embedder()
    return np.asarray(e.encode(texts), dtype="float32")


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
