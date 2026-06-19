"""
Consultation Celery Tasks
==========================
These run inside Celery worker processes, NOT the FastAPI process.
Each worker loads models once, then processes tasks from the queue.
"""

from __future__ import annotations
import asyncio
import base64
import io
import tempfile
import os
import time
import httpx
from loguru import logger
from celery import Task

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.model_registry import get_model_registry, run_onnx_classifier
from app.utils.cache import make_cache_key, sync_get_cached, sync_cache_result
from app.utils.audit import log_triage, log_soap_generated, log_transcription
from app.utils.config_loader import get_red_flags, get_specialty_map
from app.db.repositories import persist_task_result, persist_inference_metric

settings = get_settings()

_callback_client = httpx.Client(
    timeout=5.0,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)


# ── Audio acquisition (base64 or fetched URL) ─────────────────────────────────

def _validate_audio_url(url: str) -> None:
    """Guard against SSRF: http(s) only, and host allowlist if configured."""
    from urllib.parse import urlparse
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("audio_url must be http or https")
    allowed = settings.AUDIO_FETCH_ALLOWED_HOSTS
    if allowed and p.hostname not in allowed:
        raise ValueError(f"audio_url host '{p.hostname}' is not in AUDIO_FETCH_ALLOWED_HOSTS")


def _download_audio(url: str, dest_path: str) -> None:
    """Stream a URL to disk with a hard size cap (avoids loading huge files in RAM)."""
    _validate_audio_url(url)
    max_bytes = settings.AUDIO_MAX_MB * 1024 * 1024
    total = 0
    with httpx.stream("GET", url, timeout=settings.AUDIO_FETCH_TIMEOUT,
                      follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"audio exceeds AUDIO_MAX_MB ({settings.AUDIO_MAX_MB}MB)")
                f.write(chunk)
    if total == 0:
        raise ValueError("fetched audio_url is empty")


def _acquire_audio(audio_base64: str | None, audio_url: str | None,
                   audio_format: str) -> str:
    """Write audio (from base64 or a fetched URL) to a temp file; return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        if audio_url:
            _download_audio(audio_url, tmp_path)
        else:
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(audio_base64 or ""))
        return tmp_path
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _confidence(segments) -> float:
    """Rough ASR confidence: 1 − average no-speech probability."""
    if not segments:
        return 0.0
    return 1.0 - (sum(getattr(s, "no_speech_prob", 0) for s in segments) / len(segments))


def _run_whisper(whisper, audio, language):
    gen, info = whisper.transcribe(
        audio, language=language, beam_size=settings.WHISPER_BEAM_SIZE,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,
    )
    return list(gen), info


def _transcribe(whisper, tmp_path: str, language: str | None,
                diarize_stereo: bool, channel_roles: list[str] | None):
    """
    Transcribe a consultation. With diarize_stereo, split the stereo channels and
    transcribe each separately to get speaker-labeled output (left = role[0],
    right = role[1]). If the file is actually mono, fall back to a flat transcript.
    Returns (result_fields, whisper_ms).
    """
    t0 = time.perf_counter()

    if diarize_stereo:
        import numpy as np
        from faster_whisper.audio import decode_audio
        left, right = decode_audio(tmp_path, sampling_rate=16000, split_stereo=True)
        if not np.allclose(left, right):                       # genuinely stereo
            roles = (channel_roles or ["speaker_1", "speaker_2"])[:2]
            segs_out, all_segs = [], []
            lang, lang_p = None, 0.0
            for audio, role in ((left, roles[0]), (right, roles[1])):
                segs, info = _run_whisper(whisper, audio, language)
                all_segs.extend(segs)
                if lang is None:
                    lang, lang_p = info.language, info.language_probability
                for s in segs:
                    text = s.text.strip()
                    if text:
                        segs_out.append({"speaker": role, "start": round(s.start, 2),
                                         "end": round(s.end, 2), "text": text})
            segs_out.sort(key=lambda x: x["start"])
            fields = {
                "transcript": "\n".join(f"{s['speaker']}: {s['text']}" for s in segs_out),
                "segments": segs_out, "speakers": roles,
                "language_detected": lang, "language_confidence": round(lang_p, 3),
                "confidence": round(_confidence(all_segs), 3),
                "duration_seconds": round(max((s["end"] for s in segs_out), default=0.0), 2),
                "diarized": True,
            }
            return fields, (time.perf_counter() - t0) * 1000
        logger.info("diarize_stereo requested but audio is mono — flat transcript.")

    segments, info = _run_whisper(whisper, tmp_path, language)
    fields = {
        "transcript": " ".join(s.text.strip() for s in segments),
        "language_detected": info.language,
        "language_confidence": round(info.language_probability, 3),
        "confidence": round(_confidence(segments), 3),
        "duration_seconds": round(segments[-1].end if segments else 0.0, 2),
        "diarized": False,
    }
    return fields, (time.perf_counter() - t0) * 1000


class ModelTask(Task):
    """Base task class that ensures models are loaded before first task runs."""
    abstract = True
    _registry = None

    @property
    def registry(self):
        if self._registry is None:
            self._registry = get_model_registry()
            if not self._registry.is_ready:
                self._registry.load_all()
        return self._registry


# ── Red flag keywords for emergency detection ─────────────────────────────────


# ================================================================ #
# Transcription                                                      #
# ================================================================ #

@celery_app.task(
    bind=True,
    base=ModelTask,
    name="app.tasks.consultation_tasks.task_transcribe",
    max_retries=2,
)
def task_transcribe(
    self,
    task_id: str,
    session_id: str,
    audio_base64: str | None = None,
    audio_format: str = "wav",
    language: str | None = None,
    speaker: str = "patient",
    audio_url: str | None = None,
    diarize_stereo: bool = False,
    channel_roles: list[str] | None = None,
    callback_url: str | None = None,
):
    """
    Transcribe audio using faster-whisper (CTranslate2 INT8 backend).
    No pkg_resources, no system ffmpeg required — PyAV handles decoding.
    """
    cache_key = make_cache_key("transcribe", session_id)
    cached = sync_get_cached(cache_key)
    if cached:
        logger.info(f"Cache hit: transcription for session {session_id}")
        _send_callback(callback_url, task_id, cached)
        return cached

    t_start = time.perf_counter()
    try:
        tmp_path = _acquire_audio(audio_base64, audio_url, audio_format)

        try:
            fields, whisper_ms = _transcribe(self.registry.whisper, tmp_path, language,
                                             diarize_stereo, channel_roles)
        finally:
            os.unlink(tmp_path)

        result = {"success": True, "task_id": task_id, "session_id": session_id, **fields}

        sync_cache_result(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)
        log_transcription(session_id, fields["duration_seconds"], fields["language_detected"],
                          request_id=task_id)
        _send_callback(callback_url, task_id, result)

        persist_inference_metric(task_id, "whisper", whisper_ms,
                                 fields["language_detected"], fields["confidence"])
        persist_task_result(
            task_id=task_id, task_type="transcribe",
            entity_id=session_id, entity_type="session",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary={"language": fields["language_detected"],
                            "duration_seconds": fields["duration_seconds"],
                            "confidence": fields["confidence"], "diarized": fields["diarized"]},
        )
        return result

    except Exception as exc:
        persist_task_result(
            task_id=task_id, task_type="transcribe",
            entity_id=session_id, entity_type="session",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary=None, error=str(exc),
        )
        logger.error(f"Transcription task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=2)


# ================================================================ #
# Triage                                                             #
# ================================================================ #

@celery_app.task(
    bind=True,
    base=ModelTask,
    name="app.tasks.consultation_tasks.task_triage_emergency",
    max_retries=1,
    priority=10,  # Highest priority
)
def task_triage_emergency(self, task_id: str, **kwargs):
    """Emergency triage — same logic but highest queue priority."""
    return _run_triage(self, task_id, **kwargs)


@celery_app.task(
    bind=True,
    base=ModelTask,
    name="app.tasks.consultation_tasks.task_triage_normal",
    max_retries=2,
)
def task_triage_normal(self, task_id: str, **kwargs):
    return _run_triage(self, task_id, **kwargs)


def _run_triage(self_task, task_id: str, patient_id: str, symptoms: str,
                age: int | None, vital_signs: dict | None, medical_history: list,
                callback_url: str | None = None) -> dict:
    t_start = time.perf_counter()
    symptoms_lower = symptoms.lower()

    # ── Red flag check (always overrides ML score) ────────────────────────
    red_flags = [kw for kw in get_red_flags() if kw in symptoms_lower]

    # ── ML urgency scoring via ONNX ───────────────────────────────────────
    session, tokenizer = self_task.registry.triage
    input_text = f"Symptoms: {symptoms}."
    if age:
        input_text += f" Age {age}."
    if medical_history:
        input_text += f" History: {', '.join(medical_history)}."

    urgency_score = 0.5
    scores = []
    if session is not None:
        id2label = self_task.registry.get_id2label("triage")
        t_onnx = time.perf_counter()
        scores = run_onnx_classifier(session, tokenizer, input_text, id2label=id2label)
        onnx_ms = (time.perf_counter() - t_onnx) * 1000
        urgent_score = next((s["score"] for s in scores if s["label"] == "urgent"), scores[0]["score"])
        urgency_score = urgent_score
        persist_inference_metric(task_id, "triage", onnx_ms, scores[0]["label"], scores[0]["score"])

    # ── Determine urgency level ───────────────────────────────────────────
    if red_flags:
        urgency_level = "emergency"
        urgency_score = 1.0
    elif urgency_score >= settings.TRIAGE_EMERGENCY_THRESHOLD:
        urgency_level = "urgent"
    elif urgency_score >= settings.TRIAGE_URGENT_THRESHOLD:
        urgency_level = "semi_urgent"
    elif urgency_score >= 0.30:
        urgency_level = "non_urgent"
    else:
        urgency_level = "self_care"

    # ── Specialty routing ─────────────────────────────────────────────────
    specialty = "General Practitioner"
    for kw, spec in get_specialty_map().items():
        if kw in symptoms_lower:
            specialty = spec
            break

    result = {
        "success": True,
        "task_id": task_id,
        "patient_id": patient_id,
        "urgency_level": urgency_level,
        "urgency_score": round(urgency_score, 3),
        "red_flag_symptoms": red_flags,
        "recommended_specialty": specialty,
        "reasoning": (
            f"Urgency score: {urgency_score:.2f}. "
            + (f"Red flags: {', '.join(red_flags)}." if red_flags else "No red flags detected.")
        ),
        "self_care_advice": (
            "Rest, stay hydrated, monitor symptoms. Return if symptoms worsen."
            if urgency_level == "self_care" else None
        ),
    }

    log_triage(patient_id, urgency_level, red_flags, request_id=task_id)
    _send_callback(self_task.request.kwargs.get("callback_url"), task_id, result)

    persist_task_result(
        task_id=task_id, task_type="triage",
        entity_id=patient_id, entity_type="patient",
        duration_ms=int((time.perf_counter() - t_start) * 1000),
        result_summary={"urgency_score": round(urgency_score, 3), "specialty": specialty},
        urgency_level=urgency_level,
    )
    return result


# ================================================================ #
# SOAP Note                                                          #
# ================================================================ #

@celery_app.task(
    bind=True,
    base=ModelTask,
    name="app.tasks.consultation_tasks.task_soap_note",
    max_retries=2,
    time_limit=90,   # SOAP generation can take longer
)
def task_soap_note(self, task_id: str, session_id: str, transcript: str,
                   patient_id: str, doctor_id: str, specialty: str | None = None,
                   callback_url: str | None = None) -> dict:
    cache_key = make_cache_key("soap", session_id)
    cached = sync_get_cached(cache_key)
    if cached:
        _send_callback(callback_url, task_id, cached)
        return cached

    t_start = time.perf_counter()
    try:
        # ── LLM SOAP generation (folds entity extraction + grounded ICD) ───
        from app.clinical import generate_soap
        t_gen = time.perf_counter()
        gen = generate_soap(transcript, patient_id=patient_id, specialty=specialty)
        persist_inference_metric(task_id, "soap_llm", (time.perf_counter() - t_gen) * 1000)

        result = {
            "success": True,
            "task_id": task_id,
            "session_id": session_id,
            "patient_id": patient_id,
            "soap_note": gen["soap_note"],
            "extracted_entities": gen["extracted_entities"],
            "icd_suggestions": gen["icd_suggestions"],
            "llm_used": gen["llm_used"],
            "degraded": gen["degraded"],
        }

        sync_cache_result(cache_key, result)
        log_soap_generated(patient_id, session_id, icd_codes, request_id=task_id)
        _send_callback(callback_url, task_id, result)

        persist_task_result(
            task_id=task_id, task_type="soap_note",
            entity_id=patient_id, entity_type="patient",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary={"icd_codes": icd_codes, "session_id": session_id},
        )
        return result

    except Exception as exc:
        persist_task_result(
            task_id=task_id, task_type="soap_note",
            entity_id=patient_id, entity_type="patient",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary=None, error=str(exc),
        )
        logger.error(f"SOAP task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=3)


# ================================================================ #
# Patient-friendly visit summary                                     #
# ================================================================ #

@celery_app.task(
    bind=True,
    name="app.tasks.consultation_tasks.task_summary",
    max_retries=2,
    time_limit=60,
)
def task_summary(self, task_id: str, session_id: str, transcript: str,
                 callback_url: str | None = None) -> dict:
    """Plain-language visit summary for the patient (LLM, app/clinical)."""
    cache_key = make_cache_key("summary", session_id)
    cached = sync_get_cached(cache_key)
    if cached:
        _send_callback(callback_url, task_id, cached)
        return cached

    t_start = time.perf_counter()
    try:
        from app.clinical import generate_summary
        gen = generate_summary(transcript, session_id=session_id)
        result = {"success": True, "task_id": task_id, "session_id": session_id, **gen}

        if not gen["degraded"]:
            sync_cache_result(cache_key, result)
        _send_callback(callback_url, task_id, result)
        persist_task_result(
            task_id=task_id, task_type="summary",
            entity_id=session_id, entity_type="session",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary={"degraded": gen["degraded"]},
        )
        return result

    except Exception as exc:
        persist_task_result(
            task_id=task_id, task_type="summary",
            entity_id=session_id, entity_type="session",
            duration_ms=int((time.perf_counter() - t_start) * 1000),
            result_summary=None, error=str(exc),
        )
        logger.error(f"Summary task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=3)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _send_callback(callback_url: str | None, task_id: str, result: dict):
    """Fire-and-forget webhook back to Fastify with the AI result."""
    if not callback_url:
        return
    try:
        _callback_client.post(
            callback_url,
            json={"task_id": task_id, "result": result},
            headers={"X-Callback-Secret": settings.FASTIFY_CALLBACK_SECRET},
        )
    except Exception as e:
        logger.warning(f"Callback to {callback_url} failed for task {task_id}: {e}")
