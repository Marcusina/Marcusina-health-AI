"""
API Routes — FastAPI (async, non-blocking)
==========================================
Every endpoint:
1. Validates the request (Pydantic, <1ms)
2. Checks Redis cache — returns instantly on hit
3. Enqueues Celery task — returns task_id in <5ms
4. Celery worker processes async, sends result via webhook OR
   Fastify polls GET /task/{task_id} for the result

This means FastAPI NEVER blocks on AI inference.
At 50k req/sec, Fastify + FastAPI handle queueing; workers scale horizontally.
"""

from __future__ import annotations
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.core.security import verify_internal_secret
from app.core.config import get_settings
from app.models.schemas import (
    TranscribeRequest, SOAPRequest, TriageRequest,
    ModerationRequest, RecommendationRequest, SentimentRequest,
    EnqueueResponse, TaskStatusResponse,
)
from app.utils.cache import async_get_cached, make_cache_key

settings = get_settings()
router = APIRouter(dependencies=[Depends(verify_internal_secret)])


# ── Task result polling ───────────────────────────────────────────────────────

@router.get(
    "/task/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll task result (Fastify uses this for sync-style flows)",
)
async def get_task_result(task_id: str) -> TaskStatusResponse:
    """
    Fastify can poll this after receiving a task_id.
    Returns the result if ready, or status=pending if still processing.
    """
    from app.core.celery_app import celery_app
    task = celery_app.AsyncResult(task_id)

    if task.state == "SUCCESS":
        return TaskStatusResponse(task_id=task_id, status="complete", result=task.result)
    elif task.state == "FAILURE":
        return TaskStatusResponse(task_id=task_id, status="failed", error=str(task.info))
    elif task.state in ("STARTED", "RETRY"):
        return TaskStatusResponse(task_id=task_id, status="processing")
    else:
        return TaskStatusResponse(task_id=task_id, status="pending")


# ================================================================ #
# E-Consultation                                                     #
# ================================================================ #

@router.post("/consultation/transcribe", response_model=EnqueueResponse,
             summary="Transcribe consultation audio (async)")
async def transcribe(request: TranscribeRequest) -> EnqueueResponse:
    # Fast cache check before even enqueuing
    cache_key = make_cache_key("transcribe", request.session_id)
    cached = await async_get_cached(cache_key)
    if cached:
        return EnqueueResponse(task_id="cached", status="complete", result=cached)

    task_id = str(uuid.uuid4())
    from app.tasks.consultation_tasks import task_transcribe
    task_transcribe.apply_async(
        kwargs=dict(
            task_id=task_id,
            session_id=request.session_id,
            audio_base64=request.audio_base64,
            audio_format=request.audio_format,
            language=request.language,
            speaker=request.speaker,
            callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
        ),
        task_id=task_id,
    )
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

    from app.tasks.consultation_tasks import task_triage_emergency, task_triage_normal
    task_fn = task_triage_emergency if is_emergency else task_triage_normal
    task_fn.apply_async(
        kwargs=dict(
            task_id=task_id,
            patient_id=request.patient_id,
            symptoms=request.symptoms,
            age=request.age,
            vital_signs=request.vital_signs,
            medical_history=request.medical_history or [],
            callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
        ),
        task_id=task_id,
        priority=10 if is_emergency else 5,
    )
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
    from app.tasks.consultation_tasks import task_soap_note
    task_soap_note.apply_async(
        kwargs=dict(
            task_id=task_id,
            session_id=request.session_id,
            transcript=request.transcript,
            patient_id=request.patient_id,
            doctor_id=request.doctor_id,
            specialty=request.specialty,
            callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
        ),
        task_id=task_id,
    )
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
    from app.tasks.social_media_tasks import task_moderate
    task_moderate.apply_async(
        kwargs=dict(
            task_id=task_id,
            content_id=request.content_id,
            content_type=request.content_type,
            text=request.text,
            author_id=request.author_id,
            callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
        ),
        task_id=task_id,
    )
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
    from app.tasks.social_media_tasks import task_recommend
    task_recommend.apply_async(
        kwargs=dict(
            task_id=task_id,
            user_id=request.user_id,
            context=request.context,
            user_interests=request.user_interests,
            user_conditions=request.user_conditions,
            limit=request.limit,
            callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
        ),
        task_id=task_id,
    )
    return EnqueueResponse(task_id=task_id, status="queued")


@router.post("/social/sentiment", response_model=EnqueueResponse,
             summary="Sentiment + mental health concern analysis (async)")
async def sentiment(request: SentimentRequest) -> EnqueueResponse:
    task_id = str(uuid.uuid4())
    from app.tasks.social_media_tasks import task_sentiment
    task_sentiment.apply_async(
        kwargs=dict(
            task_id=task_id,
            content_id=request.content_id,
            text=request.text,
            callback_url=request.callback_url or settings.FASTIFY_CALLBACK_URL,
        ),
        task_id=task_id,
    )
    return EnqueueResponse(task_id=task_id, status="queued")
