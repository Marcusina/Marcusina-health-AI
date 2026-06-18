"""
Search & recommendation logic over the content vector store.

  * semantic_search — embed the query, return the most similar content.
  * recommend       — build a query from the user's interests/conditions/context,
                      retrieve, exclude already-seen items, and re-rank.

Both are fast, CPU-only, and synchronous. Output shapes match the API schemas.
"""

from __future__ import annotations

from app.search.store import get_store, VectorStore

SEARCH_VERSION = "search-embed/2026.06"


def semantic_search(query: str, k: int = 10, content_type: str | None = None,
                    store: VectorStore | None = None) -> dict:
    s = store or get_store()
    hits = s.search(query, k=k, type_filter=content_type)
    return {
        "query": query,
        "results": [_to_result(h) for h in hits],
        "count": len(hits),
        "model_version": SEARCH_VERSION,
    }


def recommend(interests: list[str] | None = None, conditions: list[str] | None = None,
              context: str = "", k: int = 10, exclude: list[str] | None = None,
              store: VectorStore | None = None) -> dict:
    s = store or get_store()
    exclude_set = set(exclude or [])
    profile = " ".join([*(interests or []), *(conditions or []), context]).strip()

    # No profile or empty index → trending fallback (stable sample of the catalog).
    if not profile or len(s) == 0:
        recs = [_to_result({**r, "score": None}, reason="Popular in your community")
                for r in list(s._records.values()) if r["id"] not in exclude_set][:k]
        return {"recommendations": recs, "strategy": "trending", "model_version": SEARCH_VERSION}

    hits = [h for h in s.search(profile, k=k * 2) if h["id"] not in exclude_set]

    # Light re-rank: after a consultation, prefer explanatory articles/guides.
    if context == "after_consultation":
        hits.sort(key=lambda h: h["score"] * (1.3 if h["type"] in ("article", "guide") else 1.0),
                  reverse=True)

    recs = [_to_result(h, reason=_reason(h, interests, conditions)) for h in hits[:k]]
    return {"recommendations": recs, "strategy": "content_based", "model_version": SEARCH_VERSION}


def index_content(items: list[dict], persist: bool = True, store: VectorStore | None = None) -> dict:
    s = store or get_store()
    n = s.upsert(items)
    if persist:
        s.save()
    return {"indexed": n, "total": len(s)}


def remove_content(ids: list[str], persist: bool = True, store: VectorStore | None = None) -> dict:
    s = store or get_store()
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
