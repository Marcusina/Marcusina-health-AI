"""
Content semantic search & recommendations.

Embedding-powered, CPU-only (no GPU, no API). The backend pushes content into the
store via the index endpoints — the AI service never reads the backend's database.

Public surface:
    semantic_search(query, k, content_type) -> dict
    recommend(interests, conditions, context, k, exclude) -> dict
    index_content(items) / remove_content(ids)
    get_store() -> VectorStore
"""

from app.search.service import semantic_search, recommend, index_content, remove_content
from app.search.store import get_store

__all__ = [
    "semantic_search", "recommend", "index_content", "remove_content", "get_store",
]
