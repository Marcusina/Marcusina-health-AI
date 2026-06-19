"""
Tests for content search & recommendations (app/search).

Store/ranking logic uses a fake embedder (no model load). One gated integration
test exercises the real embedder + seed content.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.search.store import VectorStore
from app.search import service


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]], dim: int = 3):
        self.mapping = mapping
        self.dim = dim

    def encode(self, texts, **kwargs):
        return np.array([self.mapping.get(t, [0.0] * self.dim) for t in texts], dtype="float32")


_MAP = {
    "diabetes diet": [1, 0, 0],
    "malaria prevention": [0, 1, 0],
    "heart exercise": [0, 0, 1],
    "diabetes": [1, 0, 0],
    "exercise": [0, 0, 1],
}
_ITEMS = [
    {"id": "a", "text": "diabetes diet", "type": "article"},
    {"id": "b", "text": "malaria prevention", "type": "guide"},
    {"id": "c", "text": "heart exercise", "type": "article"},
]


def _store() -> VectorStore:
    s = VectorStore(embedder=FakeEmbedder(_MAP))
    s.upsert(_ITEMS)
    return s


# ── store ─────────────────────────────────────────────────────────────────────

def test_upsert_and_search_ranks_by_similarity():
    s = _store()
    assert len(s) == 3
    hits = s.search("diabetes", k=2)
    assert hits[0]["id"] == "a"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)


def test_search_type_filter():
    s = _store()
    hits = s.search("exercise", k=5, type_filter="guide")
    assert all(h["type"] == "guide" for h in hits)


def test_upsert_replaces_existing():
    s = _store()
    s.upsert([{"id": "a", "text": "heart exercise", "type": "article"}])  # move 'a'
    assert len(s) == 3
    hits = s.search("exercise", k=3)
    assert "a" in [h["id"] for h in hits[:2]]


def test_remove():
    s = _store()
    assert s.remove(["b", "zzz"]) == 1
    assert len(s) == 2


def test_save_and_load_roundtrip(tmp_path):
    s = _store()
    s.save(tmp_path)
    s2 = VectorStore(embedder=FakeEmbedder(_MAP))
    assert s2.load(tmp_path) is True
    assert len(s2) == 3
    assert s2.search("diabetes", k=1)[0]["id"] == "a"


# ── multi-worker correctness ──────────────────────────────────────────────────

def test_atomic_save_leaves_no_temp_or_munged_files(tmp_path):
    _store().save(tmp_path)
    assert (tmp_path / "vectors.npy").exists() and (tmp_path / "records.jsonl").exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert not (tmp_path / "vectors.npy.tmp.npy").exists()   # np.save name-munge guard


def test_reload_if_changed_picks_up_another_workers_write(tmp_path):
    worker_a = VectorStore(embedder=FakeEmbedder(_MAP))
    worker_b = VectorStore(embedder=FakeEmbedder(_MAP))
    worker_a.upsert(_ITEMS); worker_a.save(tmp_path)
    assert worker_b.load(tmp_path) and len(worker_b) == 3

    # A indexes a new item; B shouldn't see it until it reloads.
    worker_a.upsert([{"id": "d", "text": "diabetes", "type": "article"}])
    worker_a.save(tmp_path)
    assert "d" not in worker_b._records
    assert worker_b.reload_if_changed(tmp_path) is True
    assert "d" in worker_b._records
    assert worker_b.reload_if_changed(tmp_path) is False     # unchanged → no reload


def test_concurrent_writers_do_not_clobber_each_other(tmp_path):
    # Mirrors the service write cycle (reload-latest → mutate → save). Without the
    # reload-before-write, worker B's save would overwrite worker A's item.
    worker_a = VectorStore(embedder=FakeEmbedder(_MAP))
    worker_b = VectorStore(embedder=FakeEmbedder(_MAP))

    worker_a.reload_if_changed(tmp_path); worker_a.upsert([_ITEMS[0]]); worker_a.save(tmp_path)  # 'a'
    worker_b.reload_if_changed(tmp_path); worker_b.upsert([_ITEMS[2]]); worker_b.save(tmp_path)  # 'c'

    final = VectorStore(embedder=FakeEmbedder(_MAP))
    final.load(tmp_path)
    assert set(final._records) == {"a", "c"}                 # both survived


def test_service_write_path_takes_lock_and_persists(tmp_path, monkeypatch):
    # Exercises the real index_write_lock + shared-disk persistence via the service.
    import app.search.store as store_mod
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(store_mod, "_LOCK_PATH", tmp_path / "index.lock")
    monkeypatch.setattr(store_mod, "_store", VectorStore(embedder=FakeEmbedder(_MAP)))

    out = service.index_content(_ITEMS)          # store=None → lock + reload + save
    assert out["indexed"] == 3
    assert (tmp_path / "records.jsonl").exists()

    fresh = VectorStore(embedder=FakeEmbedder(_MAP))     # a different worker
    assert fresh.load(tmp_path) and len(fresh) == 3


# ── service ───────────────────────────────────────────────────────────────────

def test_semantic_search_shape():
    out = service.semantic_search("diabetes", k=2, store=_store())
    assert out["count"] >= 1
    assert out["results"][0]["content_id"] == "a"
    assert "model_version" in out


def test_recommend_content_based_excludes_seen():
    out = service.recommend(interests=["diabetes diet"], k=3, exclude=["a"], store=_store())
    assert out["strategy"] == "content_based"
    ids = [r["content_id"] for r in out["recommendations"]]
    assert "a" not in ids


def test_recommend_trending_fallback_on_empty_profile():
    out = service.recommend(interests=[], conditions=[], context="", k=2, store=_store())
    assert out["strategy"] == "trending"
    assert len(out["recommendations"]) == 2


def test_recommend_reason_mentions_interest():
    out = service.recommend(interests=["diabetes diet"], k=1, store=_store())
    assert "diabetes diet" in out["recommendations"][0]["reason"]


# ── gated real-embedder integration ───────────────────────────────────────────

@pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"),
                    reason="loads the real embedder; set RUN_SLOW_TESTS=1 to run")
def test_real_seed_search():
    s = VectorStore()
    s.load_seed()
    hits = s.search("how do I manage my blood sugar with food", k=3)
    assert hits[0]["metadata"].get("topic") == "diabetes"
