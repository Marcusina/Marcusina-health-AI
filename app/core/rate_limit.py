"""
Redis-backed fixed-window rate limiting.

Why Redis and not an in-process counter: the service runs many uvicorn workers
(and replicas), so an in-memory limit would actually be `limit × workers`. A
shared Redis counter gives one true global limit per identity.

Identity = the `X-Client-Id` header if the caller sends one (so the backend can
rate-limit per end user), otherwise the client IP. The only auth is a shared
secret, so without X-Client-Id the practical key is the backend's IP — still
valuable as flood/backpressure protection.

Fails OPEN: if Redis is unavailable the request is allowed (with a warning). For
a safety-critical service, dropping all traffic because the limiter's backend is
down is worse than briefly not enforcing a limit.
"""

from __future__ import annotations

import time

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.utils.cache import _get_async_redis

# Never rate-limited: health/metrics probes and API docs.
DEFAULT_EXEMPT = frozenset({"/health", "/metrics", "/", "/docs", "/redoc", "/openapi.json"})


async def check_rate_limit(identity: str, limit: int, window: int) -> tuple[bool, int, int]:
    """
    Count one request for `identity` in the current fixed window.

    Returns (allowed, remaining, retry_after_seconds). Fails open (allowed=True)
    if Redis is unavailable.
    """
    redis = await _get_async_redis()
    if redis is None:
        return True, limit, 0                       # fail open

    now = int(time.time())
    bucket = now // window                          # clean per-window key, auto-expires
    key = f"health_ai:rl:{identity}:{bucket}"
    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, window + 5)     # slack so the key outlives the window
    except Exception as e:
        logger.warning(f"[ratelimit] Redis error, failing open: {e}")
        return True, limit, 0

    if count > limit:
        return False, 0, window - (now % window)
    return True, max(0, limit - count), 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, limit: int, window: int,
                 trust_forwarded: bool = False, exempt_paths=DEFAULT_EXEMPT):
        super().__init__(app)
        self.limit = limit
        self.window = window
        self.trust_forwarded = trust_forwarded
        self.exempt = set(exempt_paths)

    def _identity(self, request: Request) -> str:
        client_id = request.headers.get("x-client-id")
        if client_id:
            return f"cid:{client_id}"
        if self.trust_forwarded:
            xff = request.headers.get("x-forwarded-for")
            if xff:
                return f"ip:{xff.split(',')[0].strip()}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in self.exempt:
            return await call_next(request)

        allowed, remaining, retry_after = await check_rate_limit(
            self._identity(request), self.limit, self.window)

        if not allowed:
            logger.warning(f"[ratelimit] 429 for {self._identity(request)} on {request.url.path}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Slow down and retry."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
