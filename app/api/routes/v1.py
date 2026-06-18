from __future__ import annotations
import asyncio
import functools
import json
import uuid
from fastapi import APIRouter, Depends
from loguru import logger

from app.core.security import verify_internal_secret
from app.core.config import get_settings
from app.models.schemas import (
    TranscribeRequest, SOAPRequest, TriageRequest,
    SummaryRequest,
    ModerationRequest, RecommendationRequest, SentimentRequest, MisinfoCheckRequest,
    SupportAssistRequest,
    EnqueueResponse, TaskStatusResponse,
    TaskHistoryItem, InferenceMetricItem, AuditEventItem,
)
from app.db import repositories as db
from app.utils.cache import async_get_cached, async_cache_result, make_cache_key
from app.core.celery_app import celery_app

settings = get_settings()
router = APIRouter(dependencies=[Depends(verify_internal_secret)])

_TASK = {
    "transcribe":       "app.tasks.consultation_tasks.task_transcribe",
    "triage_emergency": "app.tasks.consultation_tasks.task_triage_emergency",
    "triage_normal":    "app.tasks.consultation_tasks.task_triage_normal",
    "soap_note":        "app.tasks.consultation_tasks.task_soap_note",
    "summary":          "app.tasks.consultation_tasks.task_summary",
    "moderate":         "app.tasks.social_media_tasks.task_moderate",
    "recommend":        "app.tasks.social_media_tasks.task_recommend",
    "sentiment":        "app.tasks.social_media_tasks.task_sentiment",
    "misinfo_check":    "app.tasks.social_media_tasks.task_misinfo_check",
    "support_assist":   "app.tasks.support_tasks.task_support_assist",
}


async def _enqueue(task_name: str, *, task_id: str, kwargs: dict, **send_kwargs):
    """Dispatch a Celery task off the event loop using send_task by name."""
    fn = functools.partial(celery_app.send_task, task_name, kwargs=kwargs, task_id=task_id, **send_kwargs)
    await asyncio.get_event_loop().run_in_executor(None, fn)


# ── Task result polling ───────────────────────────────────────────────────────

@router.get(
    "/task/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll task result (Fastify uses this for sync-style flows)",
)
async def get_task_result(task_id: str) -> TaskStatusResponse:
    from app.utils.cache import _get_async_redis
    redis = await _get_async_redis()
    if redis is not None:
        raw = await redis.get(f"celery-task-meta-{task_id}")
        if raw:
            meta = json.loads(raw)
            state = meta.get("status", "PENDING")
            if state == "SUCCESS":
                return TaskStatusResponse(task_id=task_id, status="complete", result=meta.get("result"))
            if state == "FAILURE":
                return TaskStatusResponse(task_id=task_id, status="failed", error=str(meta.get("result")))
            if state in ("STARTED", "RETRY"):
                return TaskStatusResponse(task_id=task_id, status="processing")

    return TaskStatusResponse(task_id=task_id, status="pending")


# ================================================================ #
# E-Consultation                                                     #
# ================================================================ #

@router.post("/consultation/transcribe", response_model=EnqueueResponse,
             summary="Transcribe consultation audio (async)")
async def transcribe(request: TranscribeRequest) -> EnqueueResponse:
    cache_key = make_cache_key("transcribe", request.session_id)
    cached = await async_get_cached(cache_key)
    if cached:
        return EnqueueResponse(task_id="cached", status="complete", result=cached)

    task_id = str(uuid.uuid4())
    await _enqueue(_TASK["transcribe"], task_id=task_id, kwargs=dict(
        task_id=task_id,
        session_id=request.session_id,
        audio_base64=request.audio_base64,
        audio_format=request.audio_format,
        language=request.language,
        speaker=request.speaker,
        callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
    ))
    logger.info(f"Enqueued transcription task {task_id} for session {request.session_id}")
    return EnqueueResponse(task_id=task_id, status="queued")


@router.post("/consultation/triage", response_model=EnqueueResponse,
             summary="Patient triage — emergency tasks get highest queue priority")
async def triage(request: TriageRequest) -> EnqueueResponse:
    task_id = str(uuid.uuid4())

    # Detect emergency before enqueuing so we route to the right queue
    symptoms_lower = request.symptoms.lower()
    is_emergency = any(kw in symptoms_lower for kw in [
        "chest pain", "can't breathe", "difficulty breathing",
        "seizure", "unconscious", "suicidal", "severe bleeding",
    ])

    task_name = _TASK["triage_emergency"] if is_emergency else _TASK["triage_normal"]
    await _enqueue(task_name, task_id=task_id, kwargs=dict(
        task_id=task_id,
        patient_id=request.patient_id,
        symptoms=request.symptoms,
        age=request.age,
        vital_signs=request.vital_signs,
        medical_history=request.medical_history or [],
        callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
    ), priority=10 if is_emergency else 5)
    logger.info(f"Enqueued {'EMERGENCY' if is_emergency else 'normal'} triage {task_id} for patient {request.patient_id}")
    return EnqueueResponse(task_id=task_id, status="queued", priority="emergency" if is_emergency else "normal")


@router.post("/consultation/soap-note", response_model=EnqueueResponse,
             summary="Generate SOAP note from transcript (async)")
async def soap_note(request: SOAPRequest) -> EnqueueResponse:
    cache_key = make_cache_key("soap", request.session_id)
    cached = await async_get_cached(cache_key)
    if cached:
        return EnqueueResponse(task_id="cached", status="complete", result=cached)

    task_id = str(uuid.uuid4())
    await _enqueue(_TASK["soap_note"], task_id=task_id, kwargs=dict(
        task_id=task_id,
        session_id=request.session_id,
        transcript=request.transcript,
        patient_id=request.patient_id,
        doctor_id=request.doctor_id,
        specialty=request.specialty,
        callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
    ))
    return EnqueueResponse(task_id=task_id, status="queued")


@router.post("/consultation/summary", response_model=EnqueueResponse,
             summary="Generate patient-friendly visit summary from transcript (async)")
async def summary(request: SummaryRequest) -> EnqueueResponse:
    cache_key = make_cache_key("summary", request.session_id)
    cached = await async_get_cached(cache_key)
    if cached:
        return EnqueueResponse(task_id="cached", status="complete", result=cached)

    task_id = str(uuid.uuid4())
    await _enqueue(_TASK["summary"], task_id=task_id, kwargs=dict(
        task_id=task_id,
        session_id=request.session_id,
        transcript=request.transcript,
        callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
    ))
    return EnqueueResponse(task_id=task_id, status="queued")


# ================================================================ #
# Health Social Media                                                #
# ================================================================ #

@router.post("/social/moderate", response_model=EnqueueResponse,
             summary="Moderate health content (async, ~200ms for most content)")
async def moderate(request: ModerationRequest) -> EnqueueResponse:
    cache_key = make_cache_key("moderate", request.content_id, request.text[:100])
    cached = await async_get_cached(cache_key)
    if cached:
        return EnqueueResponse(task_id="cached", status="complete", result=cached)

    task_id = str(uuid.uuid4())
    await _enqueue(_TASK["moderate"], task_id=task_id, kwargs=dict(
        task_id=task_id,
        content_id=request.content_id,
        content_type=request.content_type,
        text=request.text,
        author_id=request.author_id,
        callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
    ))
    return EnqueueResponse(task_id=task_id, status="queued")


@router.post("/social/recommend", response_model=EnqueueResponse,
             summary="Get personalised content recommendations (async)")
async def recommend(request: RecommendationRequest) -> EnqueueResponse:
    cache_key = make_cache_key("recommend", request.user_id, request.context,
                               *request.user_interests, *request.user_conditions)
    cached = await async_get_cached(cache_key)
    if cached:
        return EnqueueResponse(task_id="cached", status="complete", result=cached)

    task_id = str(uuid.uuid4())
    await _enqueue(_TASK["recommend"], task_id=task_id, kwargs=dict(
        task_id=task_id,
        user_id=request.user_id,
        context=request.context,
        user_interests=request.user_interests,
        user_conditions=request.user_conditions,
        limit=request.limit,
        callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
    ))
    return EnqueueResponse(task_id=task_id, status="queued")


@router.post("/social/sentiment", response_model=EnqueueResponse,
             summary="Sentiment + mental health concern analysis (async)")
async def sentiment(request: SentimentRequest) -> EnqueueResponse:
    cache_key = make_cache_key("sentiment", request.content_id, request.text[:100])
    cached = await async_get_cached(cache_key)
    if cached:
        return EnqueueResponse(task_id="cached", status="complete", result=cached)

    task_id = str(uuid.uuid4())
    await _enqueue(_TASK["sentiment"], task_id=task_id, kwargs=dict(
        task_id=task_id,
        content_id=request.content_id,
        text=request.text,
        callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
    ))
    return EnqueueResponse(task_id=task_id, status="queued")


@router.post("/misinfo/check", response_model=EnqueueResponse,
             summary="Grounded health-misinformation check (RAG: retrieve + LLM judge, async)")
async def misinfo_check(request: MisinfoCheckRequest) -> EnqueueResponse:
    cache_key = make_cache_key("misinfo", request.text[:120])
    cached = await async_get_cached(cache_key)
    if cached:
        return EnqueueResponse(task_id="cached", status="complete", result=cached)

    task_id = str(uuid.uuid4())
    await _enqueue(_TASK["misinfo_check"], task_id=task_id, kwargs=dict(
        task_id=task_id,
        text=request.text,
        entity_id=request.entity_id,
        k=request.k,
        callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
    ))
    return EnqueueResponse(task_id=task_id, status="queued")


@router.post("/support/assist", response_model=EnqueueResponse,
             summary="Draft a support reply + routing for a human agent (async)")
async def support_assist(request: SupportAssistRequest) -> EnqueueResponse:
    cache_key = make_cache_key("support", request.ticket_id, request.message[:100])
    cached = await async_get_cached(cache_key)
    if cached:
        return EnqueueResponse(task_id="cached", status="complete", result=cached)

    task_id = str(uuid.uuid4())
    await _enqueue(_TASK["support_assist"], task_id=task_id, kwargs=dict(
        task_id=task_id,
        ticket_id=request.ticket_id,
        subject=request.subject,
        message=request.message,
        category_hint=request.category_hint,
        callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
    ))
    return EnqueueResponse(task_id=task_id, status="queued")


# ================================================================ #
# Database query endpoints                                          #
# ================================================================ #

@router.get(
    "/history/tasks",
    response_model=list[TaskHistoryItem],
    summary="Persistent task history from PostgreSQL",
)
async def task_history(
    task_type: str | None = None,
    entity_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    return await db.get_task_history(
        task_type=task_type,
        entity_id=entity_id,
        status=status,
        limit=min(limit, 100),
        offset=offset,
    )


@router.get(
    "/history/audit",
    response_model=list[AuditEventItem],
    summary="Queryable audit log from PostgreSQL",
)
async def audit_history(
    action: str | None = None,
    entity_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    return await db.get_audit_events(
        action=action,
        entity_id=entity_id,
        limit=min(limit, 100),
        offset=offset,
    )


@router.get(
    "/metrics/inference",
    response_model=list[InferenceMetricItem],
    summary="Per-model inference latency and score stats",
)
async def inference_metrics() -> list[dict]:
    return await db.get_inference_metrics_summary()
