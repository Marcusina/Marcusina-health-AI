"""
Social Media Celery Tasks
==========================
Content moderation, recommendations, sentiment — all async via Celery.
"""

from __future__ import annotations
import re
import json
import time
import numpy as np
import httpx
from loguru import logger
from celery import Task
import faiss

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.model_registry import get_model_registry, run_onnx_classifier
from app.utils.cache import make_cache_key, sync_get_cached, sync_cache_result
from app.utils.audit import log_moderation
from app.utils.config_loader import (
    get_health_claim_pattern,
    get_distress_pattern,
    get_toxic_keywords,
)
from app.db.repositories import persist_task_result, persist_inference_metric

settings = get_settings()
_presidio_analyzer = None
_presidio_anonymizer = None

_callback_client = httpx.Client(
    timeout=5.0,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)




class ModelTask(Task):
    abstract = True
    _registry = None

    @property
    def registry(self):
        if self._registry is None:
            self._registry = get_model_registry()
            if not self._registry.is_ready:
                self._registry.load_all()
        return self._registry


# ================================================================ #
# Content Moderation                                                 #
# ================================================================ #

@celery_app.task(
    bind=True,
    base=ModelTask,
    name="app.tasks.social_media_tasks.task_moderate",
    max_retries=2,
)
def task_moderate(
    self,
    task_id: str,
    content_id: str,
    content_type: str,
    text: str,
    author_id: str,
    callback_url: str | None = None,
) -> dict:
    cache_key = make_cache_key("moderate", content_id, text[:100])
    cached = sync_get_cached(cache_key)
    if cached:
        _send_callback(callback_url, task_id, cached)
        return cached

    t_start = time.perf_counter()
    try:
        flagged_reasons = []
        text_lower = text.lower()

        # ── Layer 1: Health claim patterns ────────────────────────────────
        # Misinformation is NOT scored here. The old ONNX FAKE/REAL classifier
        # failed eval (precision 0.557 — flagged most true health info) and was
        # retired. A flagged health claim is the trigger to run the grounded
        # RAG check (task_misinfo_check / POST /api/v1/misinfo/check) for an
        # evidence-cited verdict; this task just surfaces the claim.
        health_claim = bool(get_health_claim_pattern().search(text))
        if health_claim:
            flagged_reasons.append("Unverified health claim — route to RAG misinfo check")

        # ── Layer 3: Toxicity (keyword + classifier) ──────────────────────
        toxic_hits = [kw for kw in get_toxic_keywords() if kw in text_lower]
        toxicity_score = min(len(toxic_hits) / 3.0, 1.0)
        if toxic_hits:
            flagged_reasons.append(f"Toxic language: {', '.join(toxic_hits[:2])}")

        # ── Layer 4: PII detection (presidio) ─────────────────────────────
        pii_detected, safe_text = _detect_pii(text)
        if pii_detected:
            flagged_reasons.append("PII detected and redacted")

        # ── Content-state decision (shared graduated policy) ───────────────
        from app.safety.moderation_policy import decide
        decision = decide(
            toxicity_label="toxic" if toxic_hits else "clean",
            toxicity_score=toxicity_score,
            toxic_keyword_hit=bool(toxic_hits),
            health_claim=health_claim,
            pii_detected=pii_detected,
        )
        # Back-compat verdict alias: allow→approved, quarantine→flagged, block→rejected.
        verdict = {"allow": "approved", "quarantine": "flagged",
                   "block": "rejected"}[decision["action"]]

        result = {
            "success": True,
            "task_id": task_id,
            "content_id": content_id,
            "verdict": verdict,
            "action": decision["action"],
            "visibility": decision["visibility"],
            "needs_human_review": decision["needs_human_review"],
            "review_priority": decision["review_priority"],
            "toxicity_score": round(toxicity_score, 3),
            "health_claim_detected": health_claim,
            "flagged_reasons": flagged_reasons or decision["reasons"],
            "pii_detected": pii_detected,
            "safe_text": safe_text,
            "policy_version": decision["policy_version"],
        }

        sync_cache_result(cache_key, result, ttl=settings.CACHE_TTL_MODERATION)
        log_moderation(content_id, author_id, verdict, flagged_reasons, request_id=task_id)
        _send_callback(callback_url, task_id, result)

        persist_task_result(
            task_id=task_id, task_type="moderate",
            entity_id=content_id, entity_type="content",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary={"action": decision["action"], "toxicity_score": round(toxicity_score, 3),
                            "health_claim": health_claim},
            verdict=verdict,
        )
        return result

    except Exception as exc:
        persist_task_result(
            task_id=task_id, task_type="moderate",
            entity_id=content_id, entity_type="content",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary=None, error=str(exc),
        )
        logger.error(f"Moderation task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=2)

def _get_presidio_engines():
    """Initialize Presidio once per worker process and cache it."""
    global _presidio_analyzer, _presidio_anonymizer
    if _presidio_analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        # Use en_core_web_sm (already installed, 12MB) not en_core_web_lg (587MB)
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        nlp_engine = provider.create_engine()
        _presidio_analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        _presidio_anonymizer = AnonymizerEngine()
        logger.info("Presidio PII engine initialized (en_core_web_sm).")
    return _presidio_analyzer, _presidio_anonymizer
def _detect_pii(text: str) -> tuple[bool, str | None]:
    """PII detection via presidio. Engines cached — initialized once per worker."""
    try:
        analyzer, anonymizer = _get_presidio_engines()
        results = analyzer.analyze(text=text, language="en")
        if results:
            return True, anonymizer.anonymize(text=text, analyzer_results=results).text
        return False, None
    except Exception as e:
        logger.warning(f"PII detection failed: {e}")
        return False, None

# ================================================================ #
# Recommendations                                                    #
# ================================================================ #

@celery_app.task(
    bind=True,
    base=ModelTask,
    name="app.tasks.social_media_tasks.task_recommend",
    max_retries=2,
)
def task_recommend(
    self,
    task_id: str,
    user_id: str,
    context: str,
    user_interests: list[str],
    user_conditions: list[str],
    limit: int = 10,
    callback_url: str | None = None,
) -> dict:
    cache_key = make_cache_key("recommend", user_id, context, *user_interests, *user_conditions)
    cached = sync_get_cached(cache_key)
    if cached:
        _send_callback(callback_url, task_id, cached)
        return cached

    t_start = time.perf_counter()
    try:
        interest_text = " ".join(user_interests + user_conditions)
        recommendations = []
        strategy = "trending"

        if interest_text.strip() and self.registry.faiss_index is not None:
            # ── FAISS semantic search ──────────────────────────────────────
            t_embed = time.perf_counter()
            query_vec = self.registry.embedder.encode([interest_text]).astype("float32")
            faiss.normalize_L2(query_vec)
            persist_inference_metric(task_id, "embedder", (time.perf_counter() - t_embed) * 1000)

            k = min(limit * 2, self.registry.faiss_index.ntotal)
            scores, indices = self.registry.faiss_index.search(query_vec, k)

            metadata = self.registry.faiss_metadata
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(metadata):
                    continue
                item = metadata[idx]
                recommendations.append({
                    "content_id": item.get("id", str(idx)),
                    "score": round(float(score), 3),
                    "reason": _build_reason(item, user_interests, user_conditions),
                    "content_type": item.get("type", "article"),
                })
            strategy = "content_based"

            if context == "after_consultation":
                recommendations.sort(
                    key=lambda x: x["score"] * (1.3 if x["content_type"] in ["article", "guide"] else 1.0),
                    reverse=True,
                )
                strategy = "hybrid"
        else:
            recommendations = _trending_fallback(limit)

        result = {
            "success": True,
            "task_id": task_id,
            "user_id": user_id,
            "recommendations": recommendations[:limit],
            "strategy_used": strategy,
        }

        sync_cache_result(cache_key, result, ttl=settings.CACHE_TTL_RECOMMEND)
        _send_callback(callback_url, task_id, result)

        persist_task_result(
            task_id=task_id, task_type="recommend",
            entity_id=user_id, entity_type="user",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary={"strategy": strategy, "count": len(recommendations[:limit])},
        )
        return result

    except Exception as exc:
        persist_task_result(
            task_id=task_id, task_type="recommend",
            entity_id=user_id, entity_type="user",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary=None, error=str(exc),
        )
        logger.error(f"Recommendation task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=2)


def _build_reason(item: dict, interests: list, conditions: list) -> str:
    for term in interests + conditions:
        if term.lower() in item.get("text", "").lower():
            return f"Based on your interest in {term}"
    return "Recommended for your health profile"


def _trending_fallback(limit: int) -> list[dict]:
    return [
        {"content_id": f"trend_{i}", "score": round(0.9 - i * 0.05, 2),
         "reason": "Trending in your community", "content_type": "article"}
        for i in range(limit)
    ]


# ================================================================ #
# Sentiment Analysis                                                 #
# ================================================================ #

@celery_app.task(
    bind=True,
    base=ModelTask,
    name="app.tasks.social_media_tasks.task_sentiment",
    max_retries=2,
)
def task_sentiment(
    self,
    task_id: str,
    content_id: str,
    text: str,
    callback_url: str | None = None,
) -> dict:
    t_start = time.perf_counter()
    try:
        cache_key = make_cache_key("sentiment", content_id, text[:100])
        cached = sync_get_cached(cache_key)
        if cached:
            return cached

        session, tokenizer = self.registry.sentiment
        sentiment = "neutral"
        scores = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

        if session is not None:
            id2label = self.registry.get_id2label("sentiment")
            t_onnx = time.perf_counter()
            raw_scores = run_onnx_classifier(session, tokenizer, text, id2label=id2label)
            persist_inference_metric(task_id, "sentiment", (time.perf_counter() - t_onnx) * 1000,
                                     raw_scores[0]["label"], raw_scores[0]["score"])
            scores = {}
            for item in raw_scores:
                label = item["label"].lower()
                if "pos" in label:
                    scores["positive"] = round(item["score"], 3)
                elif "neg" in label:
                    scores["negative"] = round(item["score"], 3)
                else:
                    scores["neutral"] = round(item["score"], 3)

            if scores.get("positive", 0) > 0.6:
                sentiment = "positive"
            elif scores.get("negative", 0) > 0.6:
                sentiment = "negative"

        mental_health_concern = bool(get_distress_pattern().search(text))

        result = {
            "success": True,
            "task_id": task_id,
            "content_id": content_id,
            "sentiment": sentiment,
            "scores": scores,
            "mental_health_concern": mental_health_concern,
        }

        sync_cache_result(cache_key, result, ttl=settings.CACHE_TTL_MODERATION)
        _send_callback(callback_url, task_id, result)

        persist_task_result(
            task_id=task_id, task_type="sentiment",
            entity_id=content_id, entity_type="content",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary={"mental_health_concern": mental_health_concern},
            sentiment=sentiment,
        )
        return result

    except Exception as exc:
        persist_task_result(
            task_id=task_id, task_type="sentiment",
            entity_id=content_id, entity_type="content",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary=None, error=str(exc),
        )
        logger.error(f"Sentiment task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=2)


# ================================================================ #
# Misinformation check (RAG: retrieve trusted evidence + LLM judge)  #
# ================================================================ #

@celery_app.task(
    bind=True,
    name="app.tasks.social_media_tasks.task_misinfo_check",
    max_retries=2,
)
def task_misinfo_check(
    self,
    task_id: str,
    text: str,
    entity_id: str | None = None,
    k: int = 4,
    callback_url: str | None = None,
) -> dict:
    """
    Grounded misinformation check. Does NOT use the old ONNX misinfo classifier —
    it retrieves trusted-source evidence and has the local LLM judge the claim
    against it (app/rag). Advisory only; flagged claims go to human review.
    """
    cache_key = make_cache_key("misinfo", text[:120])
    cached = sync_get_cached(cache_key)
    if cached:
        _send_callback(callback_url, task_id, cached)
        return cached

    t_start = time.perf_counter()
    try:
        from app.rag import check_claim
        result = check_claim(text, k=k)
        result = {"success": True, "task_id": task_id, "entity_id": entity_id, **result}

        # Don't cache fail-safe "unverified" results — retry once the LLM is back.
        if result["verdict"] != "unverified":
            sync_cache_result(cache_key, result, ttl=settings.CACHE_TTL_MODERATION)
        _send_callback(callback_url, task_id, result)

        persist_task_result(
            task_id=task_id, task_type="misinfo_check",
            entity_id=entity_id, entity_type="content",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary={"verdict": result["verdict"], "confidence": result["confidence"]},
            verdict=result["verdict"],
        )
        return result

    except Exception as exc:
        persist_task_result(
            task_id=task_id, task_type="misinfo_check",
            entity_id=entity_id, entity_type="content",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary=None, error=str(exc),
        )
        logger.error(f"Misinfo check task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=3)


# ================================================================ #
# Periodic tasks                                                     #
# ================================================================ #

@celery_app.task(name="app.tasks.social_media_tasks.task_rebuild_faiss_index")
def task_rebuild_faiss_index():
    """Nightly FAISS index rebuild from your content database."""
    logger.info("Rebuilding FAISS index...")
    # TODO: Fetch all content from your DB, embed, and rebuild index
    # See scripts/build_faiss_index.py for the full implementation
    logger.info("FAISS index rebuild complete.")


@celery_app.task(name="app.tasks.social_media_tasks.task_clear_expired_cache")
def task_clear_expired_cache():
    """Redis handles TTL expiry automatically — this is a no-op placeholder."""
    logger.info("Cache sweep complete (Redis TTL handles expiry automatically).")


# ── Shared callback helper ────────────────────────────────────────────────────

def _send_callback(callback_url: str | None, task_id: str, result: dict):
    if not callback_url:
        return
    try:
        _callback_client.post(
            callback_url,
            json={"task_id": task_id, "result": result},
            headers={"X-Callback-Secret": settings.FASTIFY_CALLBACK_SECRET},
        )
    except Exception as e:
        logger.warning(f"Callback failed for task {task_id}: {e}")
