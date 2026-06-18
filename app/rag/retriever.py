"""
Dense retriever over the trusted-source corpus.

Embeds the corpus once (the existing self-hosted sentence-transformer — CPU,
no GPU, no API), then ranks documents against a claim by cosine similarity. The
corpus is small, so brute-force numpy is plenty; swap in FAISS here if the corpus
grows to many thousands of documents.

The embedder is injectable so tests (and the eval harness) can run the ranking
logic without loading the model.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
from loguru import logger

from app.core.config import get_settings
from app.rag.corpus import load_corpus

settings = get_settings()


class Embedder(Protocol):
    def encode(self, texts: list[str], **kwargs) -> np.ndarray: ...


def _load_sentence_transformer() -> Embedder:
    from app.embeddings import get_embedder  # shared singleton — loaded once per process
    return get_embedder()


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Retriever:
    def __init__(self, embedder: Embedder | None = None, corpus: list[dict] | None = None):
        self._embedder = embedder
        self.corpus = corpus if corpus is not None else load_corpus()
        self._matrix: np.ndarray | None = None     # (n_docs, dim), L2-normalized

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = _load_sentence_transformer()
        return self._embedder

    def _ensure_index(self) -> None:
        if self._matrix is None:
            vecs = np.asarray(self.embedder.encode([d["text"] for d in self.corpus]),
                              dtype="float32")
            self._matrix = _l2_normalize(vecs)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-k corpus docs (each with a 'score') most similar to query."""
        if not self.corpus:
            return []
        self._ensure_index()
        q = np.asarray(self.embedder.encode([query]), dtype="float32")
        q = _l2_normalize(q)[0]
        scores = self._matrix @ q                         # cosine, since both normalized
        k = min(k, len(self.corpus))
        top = np.argsort(-scores)[:k]
        return [{**self.corpus[i], "score": round(float(scores[i]), 4)} for i in top]


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
