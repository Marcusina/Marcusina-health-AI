"""
Support Celery Tasks
====================
Async support-desk assist — drafts a reply + routing for a human agent to review.
"""

from __future__ import annotations

import time
import httpx
from loguru import logger

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.utils.cache import make_cache_key, sync_get_cached, sync_cache_result
from app.db.repositories import persist_task_result

settings = get_settings()

_callback_client = httpx.Client(
    timeout=5.0,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)


@celery_app.task(
    bind=True,
    name="app.tasks.support_tasks.task_support_assist",
    max_retries=2,
)
def task_support_assist(
    self,
    task_id: str,
    ticket_id: str,
    subject: str,
    message: str,
    category_hint: str | None = None,
    callback_url: str | None = None,
) -> dict:
    cache_key = make_cache_key("support", ticket_id, message[:100])
    cached = sync_get_cached(cache_key)
    if cached:
        _send_callback(callback_url, task_id, cached)
        return cached

    t_start = time.perf_counter()
    try:
        from app.support import draft_support_reply
        draft = draft_support_reply(subject, message, category_hint=category_hint)
        result = {"success": True, "task_id": task_id, "ticket_id": ticket_id, **draft}

        if not draft["degraded"]:
            sync_cache_result(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)
        _send_callback(callback_url, task_id, result)

        persist_task_result(
            task_id=task_id, task_type="support_assist",
            entity_id=ticket_id, entity_type="ticket",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary={"category": draft["category"], "priority": draft["priority"],
                            "distress_flag": draft["distress_flag"]},
        )
        return result

    except Exception as exc:
        persist_task_result(
            task_id=task_id, task_type="support_assist",
            entity_id=ticket_id, entity_type="ticket",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary=None, error=str(exc),
        )
        logger.error(f"Support assist task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=3)


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
        logger.warning(f"Callback failed for support task {task_id}: {e}")
