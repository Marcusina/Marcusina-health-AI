"""
Pre-consultation symptom intake assistant (#13) — advisory, NOT diagnosis.

Purpose: before a patient sees a clinician, turn their free-text complaint into a
structured intake the clinician can read at a glance, and suggest clarifying
questions to collect up front. It does **not** diagnose or recommend treatment.

Safety model:
  * The deterministic **red-flag triage net** (`app/safety.assess_triage`) is
    authoritative for urgency and routing. If it fires an emergency, the intake
    says seek immediate care — the LLM can never downgrade that.
  * The **LLM only structures and asks** — chief complaint, a tidy symptom summary,
    and clarifying questions. The system prompt forbids diagnosis/treatment.
  * **Fail safe.** If the LLM is unavailable, return the rules-based triage plus a
    minimal intake skeleton with degraded=True — never raise.

Output is always `needs_human_review=True`: a clinician owns the encounter.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from app.llm import get_llm, LLMError
from app.safety import assess_triage

INTAKE_VERSION = "symptom-intake/2026.06"

_DISCLAIMER = (
    "This is an automated intake summary to help your clinician prepare. It is NOT a "
    "diagnosis or medical advice. If your symptoms are severe or worsening, seek "
    "urgent medical care."
)

_SYS = (
    "You are a pre-consultation intake assistant for a telehealth platform. From the "
    "patient's description, organize an intake for the clinician. You MUST NOT "
    "diagnose, name conditions, suggest medications, or give treatment advice — only "
    "structure what the patient reported and ask clarifying questions a clinician would "
    "want answered.\n"
    "Produce:\n"
    "- chief_complaint: one short phrase in the patient's terms.\n"
    "- structured_summary: 1-3 sentences neutrally summarizing onset, duration, "
    "severity, and associated symptoms AS REPORTED (do not infer a cause).\n"
    "- clarifying_questions: 3-6 specific questions to ask the patient before the visit.\n"
    'Respond ONLY with JSON: {"chief_complaint": "...", "structured_summary": "...", '
    '"clarifying_questions": ["...", "..."]}.'
)


class _Intake(BaseModel):
    chief_complaint: str
    structured_summary: str
    clarifying_questions: list[str] = Field(default_factory=list)


def _prompt(symptoms: str, age: Optional[int], sex: Optional[str],
            duration: Optional[str], conditions: list[str],
            medications: list[str]) -> str:
    lines = [f"SYMPTOMS: {symptoms}"]
    if age is not None:
        lines.append(f"AGE: {age}")
    if sex:
        lines.append(f"SEX: {sex}")
    if duration:
        lines.append(f"DURATION: {duration}")
    if conditions:
        lines.append(f"EXISTING CONDITIONS: {', '.join(conditions)}")
    if medications:
        lines.append(f"CURRENT MEDICATIONS: {', '.join(medications)}")
    return "\n".join(lines)


def build_intake(symptoms: str, *, age: int | None = None, sex: str | None = None,
                 duration: str | None = None,
                 existing_conditions: list[str] | None = None,
                 medications: list[str] | None = None,
                 use_llm: bool = True) -> dict:
    """Structure a patient's complaint into an intake. Always returns (degrades if LLM down)."""
    existing_conditions = existing_conditions or []
    medications = medications or []

    # Authoritative safety net first — rules-based, GPU-independent.
    triage = assess_triage(symptoms, age=age, medical_history=existing_conditions,
                           use_llm=use_llm)
    emergency = bool(triage["red_flag_symptoms"]) or triage["urgency_level"] == "emergency"

    structured = None
    if use_llm:
        try:
            structured = get_llm().generate_json(
                messages=[{"role": "system", "content": _SYS},
                          {"role": "user", "content": _prompt(
                              symptoms, age, sex, duration, existing_conditions, medications)}],
                validate=_Intake, max_tokens=500,
            )
        except LLMError as exc:
            logger.warning(f"[intake] LLM unavailable, degraded intake: {exc}")

    if structured is not None:
        out = {
            "chief_complaint": structured.chief_complaint,
            "structured_summary": structured.structured_summary,
            "clarifying_questions": list(structured.clarifying_questions),
            "degraded": False,
        }
    else:
        # Skeleton intake — the clinician still gets the raw report + triage.
        out = {
            "chief_complaint": symptoms.strip()[:80],
            "structured_summary": symptoms.strip(),
            "clarifying_questions": [
                "When did the symptoms start and how have they changed?",
                "How severe are the symptoms (mild, moderate, severe)?",
                "Are there any other symptoms alongside this?",
            ],
            "degraded": True,
        }

    # Triage drives urgency/routing and can never be downgraded by the LLM.
    out.update({
        "urgency_level": triage["urgency_level"],
        "red_flag_symptoms": triage["red_flag_symptoms"],
        "recommended_specialty": triage["recommended_specialty"],
        "emergency": emergency,
        "patient_guidance": (
            "Your symptoms may need urgent attention — please seek immediate medical "
            "care or emergency services now." if emergency else
            "Thanks — your clinician will review this before your consultation."),
        "disclaimer": _DISCLAIMER,
        "needs_human_review": True,
        "llm_used": structured is not None,
        "model_version": INTAKE_VERSION,
    })
    return out
