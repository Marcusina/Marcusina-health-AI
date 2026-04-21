
from __future__ import annotations
import asyncio
import json
import hashlib
from typing import Optional, Any
from loguru import logger

from app.core.config import get_settings

settings = get_settings()

# ── Async client (FastAPI) ────────────────────────────────────────────────────
_async_redis = None
_async_redis_lock = asyncio.Lock()

async def _get_async_redis():
    global _async_redis
    if _async_redis is not None:
        return _async_redis
    async with _async_redis_lock:
        if _async_redis is None:
            try:
                import redis.asyncio as aioredis
                client = aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=200,
                )
                await client.ping()
                _async_redis = client
            except Exception as e:
                logger.warning(f"Async Redis unavailable: {e}")
    return _async_redis


async def async_get_cached(key: str) -> Optional[dict]:
    redis = await _get_async_redis()
    if redis is None:
        return None
    try:
        value = await redis.get(key)
        if value:
            logger.debug(f"Cache HIT: {key}")
            return json.loads(value)
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
    return None


async def async_cache_result(key: str, value: Any, ttl: int = None):
    redis = await _get_async_redis()
    if redis is None:
        return
    try:
        await redis.setex(key, ttl or settings.CACHE_TTL_SECONDS, json.dumps(value, default=str))
    except Exception as e:
        logger.warning(f"Cache set error: {e}")


# ── Sync client (Celery tasks) ────────────────────────────────────────────────
_sync_redis = None

def _get_sync_redis():
    global _sync_redis
    if _sync_redis is None:
        try:
            import redis
            _sync_redis = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
            )
            _sync_redis.ping()
        except Exception as e:
            logger.warning(f"Sync Redis unavailable: {e}")
            _sync_redis = None
    return _sync_redis


def sync_get_cached(key: str) -> Optional[dict]:
    redis = _get_sync_redis()
    if redis is None:
        return None
    try:
        value = redis.get(key)
        if value:
            return json.loads(value)
    except Exception as e:
        logger.warning(f"Sync cache get error: {e}")
    return None


def sync_cache_result(key: str, value: Any, ttl: int = None):
    redis = _get_sync_redis()
    if redis is None:
        return
    try:
        redis.setex(key, ttl or settings.CACHE_TTL_SECONDS, json.dumps(value, default=str))
    except Exception as e:
        logger.warning(f"Sync cache set error: {e}")


# ── Key builder ───────────────────────────────────────────────────────────────

def make_cache_key(prefix: str, *args) -> str:
    payload = json.dumps(args, sort_keys=True, default=str)
    digest = hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
    return f"health_ai:{prefix}:{digest}"
