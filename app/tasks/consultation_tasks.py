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
import httpx
from loguru import logger
from celery import Task

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.core.model_registry import get_model_registry, run_onnx_classifier
from app.utils.cache import make_cache_key, sync_get_cached, sync_cache_result
from app.utils.audit import log_triage, log_soap_generated, log_transcription

settings = get_settings()


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
RED_FLAGS = [
    "chest pain", "chest tightness", "can't breathe", "difficulty breathing",
    "stroke", "paralysis", "unconscious", "seizure", "severe bleeding",
    "suicidal", "overdose", "allergic reaction", "anaphylaxis",
    "heart attack", "not breathing",
]


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
    audio_base64: str,
    audio_format: str,
    language: str | None,
    speaker: str,
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

    try:
        audio_bytes = base64.b64decode(audio_base64)

        # Write to temp file — faster-whisper accepts file path or numpy array
        suffix = f".{audio_format}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # faster-whisper returns a generator of segments
            segments_gen, info = self.registry.whisper.transcribe(
                tmp_path,
                language=language,
                beam_size=settings.WHISPER_BEAM_SIZE,
                vad_filter=True,              # Skip silent segments (faster)
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=False,
            )
            segments = list(segments_gen)     # Materialise generator
        finally:
            os.unlink(tmp_path)

        transcript = " ".join(s.text.strip() for s in segments)
        duration = segments[-1].end if segments else 0.0
        confidence = 1.0 - (sum(getattr(s, "no_speech_prob", 0) for s in segments) / max(len(segments), 1))

        result = {
            "success": True,
            "task_id": task_id,
            "session_id": session_id,
            "transcript": transcript,
            "language_detected": info.language,
            "language_confidence": round(info.language_probability, 3),
            "confidence": round(confidence, 3),
            "duration_seconds": round(duration, 2),
        }

        sync_cache_result(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)
        log_transcription(session_id, duration, info.language, request_id=task_id)
        _send_callback(callback_url, task_id, result)
        return result

    except Exception as exc:
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
    symptoms_lower = symptoms.lower()

    # ── Red flag check (always overrides ML score) ────────────────────────
    red_flags = [kw for kw in RED_FLAGS if kw in symptoms_lower]

    # ── ML urgency scoring via ONNX ───────────────────────────────────────
    session, tokenizer = self_task.registry.triage
    input_text = f"Symptoms: {symptoms}."
    if age:
        input_text += f" Age {age}."
    if medical_history:
        input_text += f" History: {', '.join(medical_history)}."

    urgency_score = 0.5  # default
    if session is not None:
        scores = run_onnx_classifier(session, tokenizer, input_text)
        urgency_score = scores[0]["score"]  # Highest-scored label's confidence

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
    SPECIALTY_MAP = {
        "chest": "Pulmonologist", "cardiac": "Cardiologist",
        "mental": "Psychiatrist", "skin": "Dermatologist",
        "stomach": "Gastroenterologist", "eye": "Ophthalmologist",
        "bone": "Orthopedist", "joint": "Rheumatologist",
    }
    specialty = "General Practitioner"
    for kw, spec in SPECIALTY_MAP.items():
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

    try:
        # ── NER extraction ────────────────────────────────────────────────
        ner_session, ner_tokenizer = self.registry.ner
        entities = _extract_entities_onnx(ner_session, ner_tokenizer, transcript)

        # ── ICD code suggestions ──────────────────────────────────────────
        icd_codes = _suggest_icd(entities)

        # ── SOAP sections (rule-based extraction + NER) ───────────────────
        # For production: replace with fine-tuned clinical summariser ONNX model
        soap = _rule_based_soap(transcript, entities)

        result = {
            "success": True,
            "task_id": task_id,
            "session_id": session_id,
            "patient_id": patient_id,
            "soap_note": soap,
            "extracted_entities": entities,
            "icd_suggestions": icd_codes,
        }

        sync_cache_result(cache_key, result)
        log_soap_generated(patient_id, session_id, icd_codes, request_id=task_id)
        _send_callback(callback_url, task_id, result)
        return result

    except Exception as exc:
        logger.error(f"SOAP task {task_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=3)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_entities_onnx(session, tokenizer, text: str) -> dict:
    """NER via ONNX Runtime. Returns categorised entity dict."""
    entities = {"medications": [], "diagnoses": [], "symptoms": [], "vitals": [], "procedures": []}
    if session is None:
        return entities

    inputs = tokenizer(text[:512], return_tensors="np", truncation=True)
    ort_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
    logits = session.run(None, ort_inputs)[0][0]

    # Simplified entity extraction from token classifications
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    id2label = getattr(tokenizer, "id2label", {})

    for token, label_id in zip(tokens, logits.argmax(axis=-1)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        label = id2label.get(int(label_id), "O").upper()
        word = token.replace("##", "").strip()
        if not word or label == "O":
            continue
        if "DRUG" in label or "MED" in label:
            entities["medications"].append(word)
        elif "DISEASE" in label or "COND" in label:
            entities["diagnoses"].append(word)
        elif "SYMPTOM" in label or "SIGN" in label:
            entities["symptoms"].append(word)

    return {k: list(set(v)) for k, v in entities.items()}


def _suggest_icd(entities: dict) -> list[str]:
    ICD_MAP = {
        "hypertension": "I10", "diabetes": "E11", "asthma": "J45",
        "malaria": "B54", "typhoid": "A01.0", "pneumonia": "J18",
        "fever": "R50.9", "headache": "R51", "cough": "R05",
    }
    codes = []
    terms = entities.get("diagnoses", []) + entities.get("symptoms", [])
    for term in terms:
        for kw, code in ICD_MAP.items():
            if kw in term.lower() and code not in codes:
                codes.append(code)
    return codes[:5]


def _rule_based_soap(transcript: str, entities: dict) -> dict:
    """
    Lightweight SOAP construction from transcript + NER entities.
    Replace with a fine-tuned ONNX summariser for higher quality.
    """
    meds = ", ".join(entities.get("medications", [])) or "none noted"
    diag = ", ".join(entities.get("diagnoses", [])) or "to be determined"
    syms = ", ".join(entities.get("symptoms", [])) or "as per transcript"

    return {
        "subjective": f"Patient reports: {syms}. Transcript excerpt: {transcript[:300]}...",
        "objective": f"Extracted findings — Vitals: {entities.get('vitals', [])}. "
                     f"Procedures discussed: {entities.get('procedures', [])}.",
        "assessment": f"Probable diagnoses: {diag}.",
        "plan": f"Medications considered: {meds}. Follow-up as clinically indicated.",
    }


def _send_callback(callback_url: str | None, task_id: str, result: dict):
    """Fire-and-forget webhook back to Fastify with the AI result."""
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
        logger.warning(f"Callback to {callback_url} failed for task {task_id}: {e}")
