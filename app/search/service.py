"""
Search & recommendation logic over the content vector store.

  * semantic_search — embed the query, return the most similar content.
  * recommend       — build a query from the user's interests/conditions/context,
                      retrieve, exclude already-seen items, and re-rank.

Both are fast, CPU-only, and synchronous. Output shapes match the API schemas.
"""

from __future__ import annotations

import numpy as np

from app.embeddings import l2_normalize
from app.search.store import get_store, get_backend, index_write_lock, VectorStore

SEARCH_VERSION = "search-embed/2026.06"


def semantic_search(query: str, k: int = 10, content_type: str | None = None,
                    store: VectorStore | None = None) -> dict:
    s = store or get_store()
    if store is None:
        get_backend().reload_if_changed(s)   # pick up content other workers/hosts indexed
    hits = s.search(query, k=k, type_filter=content_type)
    return {
        "query": query,
        "results": [_to_result(h) for h in hits],
        "count": len(hits),
        "model_version": SEARCH_VERSION,
    }


def recommend(interests: list[str] | None = None, conditions: list[str] | None = None,
              context: str = "", k: int = 10, exclude: list[str] | None = None,
              seed_content_ids: list[str] | None = None,
              store: VectorStore | None = None) -> dict:
    s = store or get_store()
    if store is None:
        get_backend().reload_if_changed(s)   # pick up content other workers/hosts indexed

    seed_ids = seed_content_ids or []
    exclude_set = set(exclude or []) | set(seed_ids)   # never recommend the seeds back

    # 1. Cold-start by recent engagement — recommend content similar to what the
    #    user just interacted with. Needs NO explicit interests, and it's the
    #    strongest signal when we have it (the AI never sees the backend's DB).
    qvec = _engagement_vector(s, seed_ids)
    if qvec is not None:
        hits = [h for h in s.search_by_vector(qvec, k=k * 2) if h["id"] not in exclude_set]
        recs = [_to_result(h, reason="Because you recently viewed similar content")
                for h in _rerank(hits, context)[:k]]
        return {"recommendations": recs, "strategy": "similar_to_recent",
                "model_version": SEARCH_VERSION}

    # 2. Explicit interest/condition profile.
    profile = " ".join([*(interests or []), *(conditions or []), context]).strip()
    if profile and len(s) > 0:
        hits = [h for h in s.search(profile, k=k * 2) if h["id"] not in exclude_set]
        recs = [_to_result(h, reason=_reason(h, interests, conditions))
                for h in _rerank(hits, context)[:k]]
        return {"recommendations": recs, "strategy": "content_based",
                "model_version": SEARCH_VERSION}

    # 3. No signal at all → trending, ranked by the popularity the backend pushes.
    recs = [_to_result({**r, "score": None}, reason="Popular in your community")
            for r in s.trending(k, exclude_set)]
    return {"recommendations": recs, "strategy": "trending", "model_version": SEARCH_VERSION}


def _engagement_vector(store: VectorStore, seed_ids: list[str]):
    """Centroid of the vectors for items the user recently engaged with."""
    if not seed_ids:
        return None
    vecs = [v for v in (store.vector_of(cid) for cid in seed_ids) if v is not None]
    if not vecs:
        return None
    return l2_normalize(np.mean(np.vstack(vecs), axis=0)[None, :])[0]


def _rerank(hits: list[dict], context: str) -> list[dict]:
    # After a consultation, prefer explanatory articles/guides.
    if context == "after_consultation":
        hits = sorted(hits, key=lambda h: h["score"] * (1.3 if h["type"] in ("article", "guide") else 1.0),
                      reverse=True)
    return hits


def index_content(items: list[dict], persist: bool = True, store: VectorStore | None = None) -> dict:
    s = store or get_store()
    # For the shared index, serialize across workers/hosts and merge the latest
    # state before applying our delta, so we don't clobber another writer's content.
    if persist and store is None:
        backend = get_backend()
        with index_write_lock():
            backend.reload_if_changed(s)
            n = s.upsert(items)
            backend.persist(s)
    else:
        n = s.upsert(items)
        if persist:
            s.save()
    return {"indexed": n, "total": len(s)}


def remove_content(ids: list[str], persist: bool = True, store: VectorStore | None = None) -> dict:
    s = store or get_store()
    if persist and store is None:
        backend = get_backend()
        with index_write_lock():
            backend.reload_if_changed(s)
            n = s.remove(ids)
            backend.persist(s)
    else:
        n = s.remove(ids)
        if persist:
            s.save()
    return {"removed": n, "total": len(s)}


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_result(rec: dict, reason: str | None = None) -> dict:
    meta = rec.get("metadata", {})
    return {
        "content_id": rec["id"],
        "type": rec.get("type", "content"),
        "title": meta.get("title"),
        "score": rec.get("score"),
        "reason": reason,
        "metadata": meta,
    }


def _reason(hit: dict, interests: list[str] | None, conditions: list[str] | None) -> str:
    text = hit.get("text", "").lower()
    for term in [*(interests or []), *(conditions or [])]:
        if term and term.lower() in text:
            return f"Based on your interest in {term}"
    return "Recommended for your health profile"
