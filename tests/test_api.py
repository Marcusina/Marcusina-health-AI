"""
Test Suite — Health AI Service v2
Run: pytest tests/ -v --asyncio-mode=auto
"""

import pytest
import base64
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport

# Mock heavy deps before app import
import sys
for mod in ["faster_whisper", "onnxruntime", "faiss", "sentence_transformers",
            "transformers", "presidio_analyzer", "presidio_anonymizer", "spacy"]:
    sys.modules[mod] = MagicMock()

from app.main import app
from app.core.config import get_settings

settings = get_settings()
HEADERS = {"X-AI-Secret": settings.API_SECRET_KEY, "Content-Type": "application/json"}
BASE = "http://test"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE) as c:
        yield c


# ── Health check ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Auth ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_secret_returns_401(client):
    r = await client.post("/api/v1/social/sentiment",
                          json={"content_id": "c1", "text": "hello"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_wrong_secret_returns_403(client):
    r = await client.post("/api/v1/social/sentiment",
                          headers={"X-AI-Secret": "wrong"},
                          json={"content_id": "c1", "text": "hello"})
    assert r.status_code == 403


# ── Triage — enqueue ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_triage_enqueue_returns_task_id(client):
    with patch("app.api.routes.v1.task_triage_normal") as mock_task:
        mock_task.apply_async = MagicMock()
        r = await client.post(
            "/api/v1/consultation/triage",
            headers=HEADERS,
            json={"patient_id": "p001", "symptoms": "mild headache for two days"},
        )
    assert r.status_code == 200
    data = r.json()
    assert "task_id" in data
    assert data["status"] in ("queued", "complete")


@pytest.mark.asyncio
async def test_emergency_triage_routes_to_emergency_queue(client):
    with patch("app.api.routes.v1.task_triage_emergency") as mock_emergency, \
         patch("app.api.routes.v1.task_triage_normal") as mock_normal:
        mock_emergency.apply_async = MagicMock()
        mock_normal.apply_async = MagicMock()

        r = await client.post(
            "/api/v1/consultation/triage",
            headers=HEADERS,
            json={"patient_id": "p002", "symptoms": "severe chest pain and difficulty breathing"},
        )

    assert r.status_code == 200
    data = r.json()
    assert data.get("priority") == "emergency"
    mock_emergency.apply_async.assert_called_once()
    mock_normal.apply_async.assert_not_called()


# ── Moderation — enqueue ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_moderation_enqueue(client):
    with patch("app.api.routes.v1.task_moderate") as mock_task, \
         patch("app.utils.cache.async_get_cached", new_callable=AsyncMock, return_value=None):
        mock_task.apply_async = MagicMock()
        r = await client.post(
            "/api/v1/social/moderate",
            headers=HEADERS,
            json={
                "content_id": "post_001",
                "content_type": "post",
                "text": "Drinking water is important for kidney health.",
                "author_id": "user_123",
            },
        )
    assert r.status_code == 200
    assert r.json()["status"] in ("queued", "complete")


@pytest.mark.asyncio
async def test_moderation_cache_hit_returns_immediately(client):
    cached_result = {
        "content_id": "post_001", "verdict": "approved",
        "misinformation_score": 0.02, "toxicity_score": 0.0,
        "health_claim_detected": False, "flagged_reasons": [],
        "pii_detected": False, "safe_text": None,
    }
    with patch("app.utils.cache.async_get_cached", new_callable=AsyncMock, return_value=cached_result):
        r = await client.post(
            "/api/v1/social/moderate",
            headers=HEADERS,
            json={
                "content_id": "post_001",
                "content_type": "post",
                "text": "Drinking water is important for kidney health.",
                "author_id": "user_123",
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert data["result"]["verdict"] == "approved"


# ── Task polling ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_poll_pending(client):
    with patch("app.api.routes.v1.celery_app") as mock_celery:
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_celery.AsyncResult.return_value = mock_result

        r = await client.get("/api/v1/task/some-task-id", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_task_poll_complete(client):
    with patch("app.api.routes.v1.celery_app") as mock_celery:
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"verdict": "approved", "task_id": "some-task-id"}
        mock_celery.AsyncResult.return_value = mock_result

        r = await client.get("/api/v1/task/some-task-id", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "complete"
    assert data["result"]["verdict"] == "approved"


# ── Celery task unit tests ────────────────────────────────────────────────────

def test_triage_red_flag_detection():
    """Triage task: red flags must always produce emergency urgency."""
    from app.tasks.consultation_tasks import _run_triage

    class FakeTask:
        class registry:
            @staticmethod
            def triage():
                return None, None
        request = MagicMock()
        request.kwargs = {}

    result = _run_triage(
        FakeTask(),
        task_id="t001",
        patient_id="p001",
        symptoms="I have severe chest pain and difficulty breathing",
        age=55,
        vital_signs=None,
        medical_history=["hypertension"],
        callback_url=None,
    )
    assert result["urgency_level"] == "emergency"
    assert len(result["red_flag_symptoms"]) > 0


def test_moderation_misinformation_pattern():
    """Moderation task: health claim patterns must be detected."""
    import re
    from app.tasks.social_media_tasks import HEALTH_CLAIM_RE

    assert HEALTH_CLAIM_RE.search("This miracle cure 100% effective for cancer")
    assert HEALTH_CLAIM_RE.search("Doctors don't want you to know this")
    assert not HEALTH_CLAIM_RE.search("Drinking water daily keeps you healthy")


def test_distress_pattern_detection():
    """Sentiment task: mental health distress signals must be caught."""
    from app.tasks.social_media_tasks import DISTRESS_RE

    assert DISTRESS_RE.search("I want to die, I can't take this anymore")
    assert DISTRESS_RE.search("suicidal thoughts won't stop")
    assert not DISTRESS_RE.search("I am feeling much better today")
