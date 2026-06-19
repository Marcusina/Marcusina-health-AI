"""
Tests for the Redis-backed rate limiter (app/core/rate_limit).

A fake async Redis stands in for the real one — no network. Covers the counting
logic, fail-open behaviour, per-identity isolation, the 429 response, and that
health/metrics paths are exempt.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.core.rate_limit as rl
from app.core.rate_limit import check_rate_limit, RateLimitMiddleware


class FakeRedis:
    def __init__(self):
        self.store: dict[str, int] = {}

    async def incr(self, key):
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    async def expire(self, key, ttl):
        return True


def _use_fake(monkeypatch, redis):
    async def _getter():
        return redis
    monkeypatch.setattr(rl, "_get_async_redis", _getter)


# ── counting logic ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_allows_up_to_limit_then_blocks(monkeypatch):
    _use_fake(monkeypatch, FakeRedis())
    for _ in range(3):
        allowed, remaining, retry = await check_rate_limit("ip:a", limit=3, window=60)
        assert allowed
    allowed, remaining, retry = await check_rate_limit("ip:a", limit=3, window=60)
    assert allowed is False
    assert remaining == 0 and retry > 0


@pytest.mark.asyncio
async def test_identities_are_isolated(monkeypatch):
    _use_fake(monkeypatch, FakeRedis())
    for _ in range(3):
        await check_rate_limit("ip:a", limit=3, window=60)
    # a is now exhausted; b is independent
    assert (await check_rate_limit("ip:a", limit=3, window=60))[0] is False
    assert (await check_rate_limit("ip:b", limit=3, window=60))[0] is True


@pytest.mark.asyncio
async def test_fails_open_when_redis_down(monkeypatch):
    async def _none():
        return None
    monkeypatch.setattr(rl, "_get_async_redis", _none)
    allowed, remaining, retry = await check_rate_limit("ip:a", limit=1, window=60)
    assert allowed is True and retry == 0


# ── middleware ────────────────────────────────────────────────────────────────

def _app(monkeypatch, **kw):
    _use_fake(monkeypatch, FakeRedis())
    application = FastAPI()
    application.add_middleware(RateLimitMiddleware, limit=2, window=60, **kw)

    @application.get("/ping")
    def ping():
        return {"ok": True}

    @application.get("/health")
    def health():
        return {"status": "ok"}

    return TestClient(application)


def test_middleware_returns_429_after_limit(monkeypatch):
    client = _app(monkeypatch)
    assert client.get("/ping").status_code == 200
    r2 = client.get("/ping")
    assert r2.status_code == 200
    assert r2.headers["X-RateLimit-Remaining"] == "0"
    blocked = client.get("/ping")
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"]
    assert blocked.json()["detail"].startswith("Rate limit")


def test_middleware_exempts_health(monkeypatch):
    client = _app(monkeypatch)
    # hammer past the limit on an exempt path — never blocked
    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_middleware_scopes_by_client_id_header(monkeypatch):
    client = _app(monkeypatch)
    # different X-Client-Id values get independent budgets
    assert client.get("/ping", headers={"X-Client-Id": "u1"}).status_code == 200
    assert client.get("/ping", headers={"X-Client-Id": "u1"}).status_code == 200
    assert client.get("/ping", headers={"X-Client-Id": "u1"}).status_code == 429
    assert client.get("/ping", headers={"X-Client-Id": "u2"}).status_code == 200
