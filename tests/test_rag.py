"""
Tests for the RAG misinfo lane (app/rag).

Retrieval ranking is tested with a fake embedder (no model load). The judge/verdict
logic is tested with the LLM stubbed, including the fail-safe path. One optional
integration test exercises the real embedder if it's available.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.retriever import Retriever
from app.rag.misinfo import check_claim, _Verdict


# ── fakes ─────────────────────────────────────────────────────────────────────

class FakeEmbedder:
    """Returns a fixed vector per known text; orthogonal one-hots so cosine is clean."""
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping
        self.dim = len(next(iter(mapping.values())))

    def encode(self, texts, **kwargs):
        return np.array([self.mapping.get(t, [0.0] * self.dim) for t in texts], dtype="float32")


def _fake_retriever_returning(evidence: list[dict]):
    class _R:
        def search(self, query, k=4):
            return evidence[:k]
    return _R()


# ── retriever ranking ─────────────────────────────────────────────────────────

def test_retriever_ranks_by_cosine():
    corpus = [
        {"id": "a", "text": "alpha", "source": "X", "url": "u"},
        {"id": "b", "text": "beta", "source": "X", "url": "u"},
        {"id": "c", "text": "gamma", "source": "X", "url": "u"},
    ]
    emb = FakeEmbedder({
        "alpha": [1, 0, 0], "beta": [0, 1, 0], "gamma": [0, 0, 1],
        "find beta": [0, 1, 0],
    })
    r = Retriever(embedder=emb, corpus=corpus)
    hits = r.search("find beta", k=2)
    assert hits[0]["id"] == "b"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert len(hits) == 2


def test_retriever_empty_corpus_returns_nothing():
    r = Retriever(embedder=FakeEmbedder({"q": [1.0]}), corpus=[])
    assert r.search("q") == []


# ── check_claim verdict logic (LLM stubbed) ───────────────────────────────────

_EVIDENCE = [
    {"id": "e1", "source": "WHO", "url": "http://who", "text": "Vaccines do not cause autism.", "score": 0.9},
    {"id": "e2", "source": "CDC", "url": "http://cdc", "text": "MMR is safe and effective.", "score": 0.8},
]


def test_check_claim_contradicted_flags_for_review(monkeypatch):
    monkeypatch.setattr("app.rag.misinfo._judge",
        lambda claim, ev: _Verdict(verdict="contradicted", confidence=0.95,
                                   rationale="evidence contradicts", citation_ids=[1]))
    out = check_claim("vaccines cause autism", retriever=_fake_retriever_returning(_EVIDENCE))
    assert out["verdict"] == "contradicted"
    assert out["flag"] is True
    assert out["needs_human_review"] is True
    assert out["citations"][0]["source"] == "WHO"
    assert out["llm_used"] is True


def test_check_claim_supported_is_not_flagged(monkeypatch):
    monkeypatch.setattr("app.rag.misinfo._judge",
        lambda claim, ev: _Verdict(verdict="supported", confidence=0.9,
                                   rationale="supported", citation_ids=[2]))
    out = check_claim("the mmr vaccine is safe", retriever=_fake_retriever_returning(_EVIDENCE))
    assert out["verdict"] == "supported"
    assert out["flag"] is False
    assert out["needs_human_review"] is False
    assert out["citations"][0]["source"] == "CDC"


def test_check_claim_not_health_claim(monkeypatch):
    monkeypatch.setattr("app.rag.misinfo._judge",
        lambda claim, ev: _Verdict(verdict="not_health_claim", confidence=0.5,
                                   rationale="not a claim", citation_ids=[]))
    out = check_claim("what a nice day", retriever=_fake_retriever_returning(_EVIDENCE))
    assert out["verdict"] == "not_health_claim"
    assert out["flag"] is False


def test_check_claim_fails_safe_when_llm_down(monkeypatch):
    monkeypatch.setattr("app.rag.misinfo._judge", lambda claim, ev: None)
    out = check_claim("some health claim", retriever=_fake_retriever_returning(_EVIDENCE))
    assert out["verdict"] == "unverified"
    assert out["needs_human_review"] is True
    assert out["llm_used"] is False
    # the reviewer still gets the retrieved evidence as candidate citations
    assert len(out["citations"]) == 2


def test_check_claim_no_evidence_is_unsupported(monkeypatch):
    monkeypatch.setattr("app.rag.misinfo._judge",
        lambda claim, ev: pytest.fail("judge called with no evidence"))
    out = check_claim("obscure claim", retriever=_fake_retriever_returning([]))
    assert out["verdict"] == "unsupported"
    assert out["needs_human_review"] is True
    assert out["citations"] == []


# ── optional: real embedder integration ───────────────────────────────────────

@pytest.mark.skipif(not __import__("os").environ.get("RUN_SLOW_TESTS"),
                    reason="loads the real embedder (~80s); set RUN_SLOW_TESTS=1 to run")
def test_real_retriever_finds_relevant_topic():
    """Loads the real sentence-transformer if available; skips otherwise."""
    try:
        r = Retriever()                       # real embedder + real corpus
        hits = r.search("vaccines cause autism in children", k=3)
    except Exception as e:                     # model not downloaded / offline
        pytest.skip(f"real embedder unavailable: {e}")
    topics = {h["topic"] for h in hits}
    assert "vaccines" in topics
