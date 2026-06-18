"""
Patient-friendly visit summary from a consultation transcript.

Plain-language, supportive, and faithful to the transcript — written for the
patient, not the chart. Always carries a disclaimer and is clinician-reviewable.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from app.llm import get_llm, LLMError

SUMMARY_VERSION = "summary-llm/2026.06"

_DISCLAIMER = ("This summary is for your information and does not replace medical "
               "advice. Contact your clinician or emergency services if you feel worse.")

_SYS = (
    "You are a patient-communication assistant for a telehealth platform. Write a "
    "clear, plain-language summary of the consultation for the PATIENT to read "
    "(about a 6th-grade reading level, warm and supportive). Use ONLY what is in the "
    "transcript — do not add advice or diagnoses that were not discussed. Avoid "
    "jargon. Respond ONLY with JSON of this shape:\n"
    '{"summary": "<short plain-language paragraph>", '
    '"next_steps": ["..."], "when_to_seek_help": ["..."]}'
)


class VisitSummary(BaseModel):
    summary: str
    next_steps: list[str] = Field(default_factory=list)
    when_to_seek_help: list[str] = Field(default_factory=list)


def generate_summary(transcript: str, *, session_id: str | None = None) -> dict:
    """Generate a patient-friendly summary; degrades gracefully if the LLM is down."""
    try:
        s: VisitSummary = get_llm().generate_json(
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": f"Transcript:\n{transcript.strip()}"}],
            validate=VisitSummary, max_tokens=600,
        )
        return {
            "summary": s.summary,
            "next_steps": s.next_steps,
            "when_to_seek_help": s.when_to_seek_help,
            "disclaimer": _DISCLAIMER,
            "llm_used": True, "degraded": False, "model_version": SUMMARY_VERSION,
        }
    except LLMError as exc:
        logger.warning(f"[clinical] summary LLM unavailable, degraded: {exc}")
        return {
            "summary": "A summary of your visit could not be generated automatically. "
                       "Your clinician's notes remain the source of truth.",
            "next_steps": [], "when_to_seek_help": [],
            "disclaimer": _DISCLAIMER,
            "llm_used": False, "degraded": True, "model_version": SUMMARY_VERSION,
        }
