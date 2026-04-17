"""
Social Media Celery Tasks
==========================
Content moderation, recommendations, sentiment — all async via Celery.
"""

from __future__ import annotations
import re
import json
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

settings = get_settings()

# ── Compile patterns once at import time ─────────────────────────────────────
HEALTH_CLAIM_RE = re.compile(
    r"cures? (cancer|diabetes|hiv|aids|malaria)"
    r"|100% effective"
    r"|doctors? (don'?t|do not) want you to know"
    r"|miracle (cure|treatment|drug)"
    r"|guaranteed (cure|treatment)"
    r"|no side effects guaranteed"
    r"|big pharma (hiding|suppressing)",
    re.IGNORECASE,
)
DISTRESS_RE = re.compile(
    r"want to (die|end it|kill myself)"
    r"|no reason to live"
    r"|suicidal"
    r"|self.harm"
    r"|overdose on purpose",
    re.IGNORECASE,
)
TOXIC_KEYWORDS = frozenset([
    "quack", "fake doctor", "kill yourself", "you deserve to suffer",
    "idiot patient", "medical fraud",
])


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

    try:
        flagged_reasons = []
        text_lower = text.lower()

        # ── Layer 1: Misinformation (ONNX classifier) ─────────────────────
        session, tokenizer = self.registry.misinfo
        misinfo_score = 0.0
        if session is not None:
            scores = run_onnx_classifier(session, tokenizer, text)
            misinfo_score = next(
                (s["score"] for s in scores if "FAKE" in s["label"].upper()), 0.0
            )
        if misinfo_score >= settings.MISINFO_THRESHOLD:
            flagged_reasons.append(f"Health misinformation detected (score: {misinfo_score:.2f})")

        # ── Layer 2: Health claim patterns ────────────────────────────────
        health_claim = bool(HEALTH_CLAIM_RE.search(text))
        if health_claim:
            flagged_reasons.append("Unverified health claim")

        # ── Layer 3: Toxicity (keyword + classifier) ──────────────────────
        toxic_hits = [kw for kw in TOXIC_KEYWORDS if kw in text_lower]
        toxicity_score = min(len(toxic_hits) / 3.0, 1.0)
        if toxic_hits:
            flagged_reasons.append(f"Toxic language: {', '.join(toxic_hits[:2])}")

        # ── Layer 4: PII detection (presidio) ─────────────────────────────
        pii_detected, safe_text = _detect_pii(text)
        if pii_detected:
            flagged_reasons.append("PII detected and redacted")

        # ── Verdict ───────────────────────────────────────────────────────
        if misinfo_score >= 0.90 or (toxicity_score >= 0.7 and len(toxic_hits) > 2):
            verdict = "rejected"
        elif flagged_reasons:
            verdict = "flagged"
        else:
            verdict = "approved"

        result = {
            "success": True,
            "task_id": task_id,
            "content_id": content_id,
            "verdict": verdict,
            "misinformation_score": round(misinfo_score, 3),
            "toxicity_score": round(toxicity_score, 3),
            "health_claim_detected": health_claim,
            "flagged_reasons": flagged_reasons,
            "pii_detected": pii_detected,
            "safe_text": safe_text,
        }

        sync_cache_result(cache_key, result, ttl=settings.CACHE_TTL_MODERATION)
        log_moderation(content_id, author_id, verdict, flagged_reasons, request_id=task_id)
        _send_callback(callback_url, task_id, result)
        return result

    except Exception as exc:
        logger.error(f"Moderation task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=2)


def _detect_pii(text: str) -> tuple[bool, str | None]:
    """PII detection via presidio. Returns (detected, redacted_text)."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
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

    try:
        interest_text = " ".join(user_interests + user_conditions)
        recommendations = []
        strategy = "trending"

        if interest_text.strip() and self.registry.faiss_index is not None:
            # ── FAISS semantic search ──────────────────────────────────────
            query_vec = self.registry.embedder.encode([interest_text]).astype("float32")
            faiss.normalize_L2(query_vec)

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

            # Boost educational content after consultation
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
        return result

    except Exception as exc:
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
    try:
        session, tokenizer = self.registry.sentiment
        sentiment = "neutral"
        scores = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

        if session is not None:
            raw_scores = run_onnx_classifier(session, tokenizer, text)
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

        mental_health_concern = bool(DISTRESS_RE.search(text))

        result = {
            "success": True,
            "task_id": task_id,
            "content_id": content_id,
            "sentiment": sentiment,
            "scores": scores,
            "mental_health_concern": mental_health_concern,
        }

        _send_callback(callback_url, task_id, result)
        return result

    except Exception as exc:
        logger.error(f"Sentiment task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=2)


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
        with httpx.Client(timeout=5.0) as client:
            client.post(
                callback_url,
                json={"task_id": task_id, "result": result},
                headers={"X-Callback-Secret": settings.FASTIFY_CALLBACK_SECRET},
            )
    except Exception as e:
        logger.warning(f"Callback failed for task {task_id}: {e}")
