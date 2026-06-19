"""
Semantic search & recommendation endpoints (sync).

Fast, CPU-only (embed + cosine), no GPU and no LLM. Plain `def` handlers run in
FastAPI's threadpool so the (brief) embedding work never blocks the event loop.

The backend populates the content index via /content/index and /content/remove —
the AI service never reaches into the backend's database.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import verify_internal_secret
from app.models.schemas import (
    SearchRequest, SearchResponse, RecommendQuery, RecommendResponse,
    IndexContentRequest, RemoveContentRequest, IndexResponse,
)
from app.search import semantic_search, recommend, index_content, remove_content

router = APIRouter(dependencies=[Depends(verify_internal_secret)], tags=["Search & recommend"])


@router.post("/search", response_model=SearchResponse,
             summary="Semantic content search (embedding-based, sync)")
def search(req: SearchRequest) -> SearchResponse:
    return SearchResponse(**semantic_search(req.query, k=req.k, content_type=req.content_type))


@router.post("/recommend", response_model=RecommendResponse,
             summary="Personalized content recommendations (embedding-based, sync)")
def recommend_content(req: RecommendQuery) -> RecommendResponse:
    return RecommendResponse(**recommend(
        interests=req.user_interests, conditions=req.user_conditions,
        seed_content_ids=req.seed_content_ids,
        context=req.context, k=req.k, exclude=req.exclude,
    ))


@router.post("/content/index", response_model=IndexResponse,
             summary="Upsert content into the search/recommend index")
def content_index(req: IndexContentRequest) -> IndexResponse:
    return IndexResponse(**index_content([i.model_dump() for i in req.items]))


@router.post("/content/remove", response_model=IndexResponse,
             summary="Remove content from the index by id")
def content_remove(req: RemoveContentRequest) -> IndexResponse:
    return IndexResponse(**remove_content(req.ids))
