
from __future__ import annotations
import asyncio
import json
import hashlib
import time
from typing import Optional, Any
from loguru import logger

from app.core.config import get_settings

settings = get_settings()

# Connect/read timeouts so a down or filtered Redis fails fast instead of hanging
# the request. After a failure we back off for this long before retrying — the
# rate limiter and cache run on every request, so reconnecting each time during a
# Redis outage would stall the whole service.
_REDIS_CONNECT_TIMEOUT = 0.5
_REDIS_SOCKET_TIMEOUT = 1.0
_REDIS_DOWN_BACKOFF = 10.0

# ── Async client (FastAPI) ────────────────────────────────────────────────────
_async_redis = None
_async_redis_lock = asyncio.Lock()
_async_redis_down_until = 0.0       # skip reconnect attempts until this monotonic time

async def _get_async_redis():
    global _async_redis, _async_redis_down_until
    if _async_redis is not None:
        return _async_redis
    if time.monotonic() < _async_redis_down_until:
        return None                 # recently failed — fail fast, don't reconnect every call
    async with _async_redis_lock:
        if _async_redis is None and time.monotonic() >= _async_redis_down_until:
            try:
                import redis.asyncio as aioredis
                client = aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=200,
                    socket_connect_timeout=_REDIS_CONNECT_TIMEOUT,
                    socket_timeout=_REDIS_SOCKET_TIMEOUT,
                )
                await client.ping()
                _async_redis = client
            except Exception as e:
                logger.warning(f"Async Redis unavailable (backing off {_REDIS_DOWN_BACKOFF}s): {e}")
                _async_redis_down_until = time.monotonic() + _REDIS_DOWN_BACKOFF
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
                socket_connect_timeout=_REDIS_CONNECT_TIMEOUT,
                socket_timeout=_REDIS_SOCKET_TIMEOUT,
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
