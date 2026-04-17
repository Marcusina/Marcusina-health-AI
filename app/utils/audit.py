"""
Audit Logger — NDPR/HIPAA-compliant structured logging.
Every AI decision that touches patient data is logged to audit.log (NDJSON).
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

Path("logs").mkdir(exist_ok=True)

logger.add(
    "logs/audit.log",
    format="{message}",
    filter=lambda r: r["extra"].get("audit") is True,
    rotation="100 MB",
    retention="365 days",
    compression="gz",
)


def _write(action: str, module: str, **kwargs):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_id": str(uuid.uuid4()),
        "action": action,
        "module": module,
        **{k: v for k, v in kwargs.items() if v is not None},
    }
    logger.bind(audit=True).info(json.dumps(entry))


def log_triage(patient_id: str, urgency_level: str, red_flags: list, request_id: str = None):
    _write("triage_assessment", "consultation",
           patient_id=patient_id,
           urgency_level=urgency_level,
           red_flags=red_flags,
           request_id=request_id)


def log_soap_generated(patient_id: str, session_id: str, icd_codes: list, request_id: str = None):
    _write("soap_note_generated", "consultation",
           patient_id=patient_id,
           session_id=session_id,
           icd_suggestions=icd_codes,
           request_id=request_id)


def log_moderation(content_id: str, author_id: str, verdict: str, reasons: list, request_id: str = None):
    _write("content_moderation", "social_media",
           content_id=content_id,
           author_id=author_id,
           verdict=verdict,
           reasons=reasons,
           request_id=request_id)


def log_transcription(session_id: str, duration: float, language: str, request_id: str = None):
    _write("audio_transcribed", "consultation",
           session_id=session_id,
           duration_seconds=duration,
           language=language,
           request_id=request_id)
