"""
LLM escalation for ambiguous safety cases.

Each judge asks the local LLM for a strict, schema-validated opinion and returns
it — or returns ``None`` if the LLM is unavailable or misbehaves. Callers treat
``None`` as "no opinion" and fall back to the deterministic rules, choosing the
**safe** default (over-escalate for self-harm, don't silently downgrade triage).

These run on the on-demand GPU lane, so they must never hard-fail the request.
"""

from __future__ import annotations

from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel

from app.llm import get_llm, LLMError


# ── schemas the model must fill ───────────────────────────────────────────────

class DistressJudgement(BaseModel):
    severity: Literal["none", "low", "high"]
    escalate: bool
    rationale: str


class ToxicityJudgement(BaseModel):
    label: Literal["clean", "toxic", "harassment"]
    action: Literal["allow", "flag", "block"]
    rationale: str


class TriageJudgement(BaseModel):
    urgency_level: Literal["emergency", "urgent", "semi_urgent", "non_urgent", "self_care"]
    rationale: str


# ── judges (each returns the model or None) ───────────────────────────────────

_DISTRESS_SYS = (
    "You are a clinical safety assistant for a telehealth platform. Assess whether "
    "a user's message indicates risk of self-harm or suicide. Distinguish genuine "
    "ideation/hopelessness/planning from benign figurative language (e.g. 'this "
    "deadline is killing me', 'I'm dead tired today'). Err toward caution: when "
    "genuinely unsure, escalate. Respond ONLY with JSON: "
    '{"severity": "none|low|high", "escalate": true|false, "rationale": "<one sentence>"}.'
)


def judge_distress(text: str) -> Optional[DistressJudgement]:
    try:
        return get_llm().generate_json(
            messages=[
                {"role": "system", "content": _DISTRESS_SYS},
                {"role": "user", "content": text},
            ],
            validate=DistressJudgement,
            max_tokens=200,
        )
    except LLMError as exc:
        logger.warning(f"[safety] distress judge unavailable, failing safe: {exc}")
        return None


_TOXICITY_SYS = (
    "You are a content moderator for a health community platform. Classify the "
    "message for toxicity, harassment, or harmful content (bullying, medical "
    "shaming, quackery accusations, slurs, threats). Be precision-oriented: do not "
    "flag legitimate medical disagreement or frank health discussion. Respond ONLY "
    'with JSON: {"label": "clean|toxic|harassment", '
    '"action": "allow|flag|block", "rationale": "<one sentence>"}.'
)


def judge_toxicity(text: str) -> Optional[ToxicityJudgement]:
    try:
        return get_llm().generate_json(
            messages=[
                {"role": "system", "content": _TOXICITY_SYS},
                {"role": "user", "content": text},
            ],
            validate=ToxicityJudgement,
            max_tokens=200,
        )
    except LLMError as exc:
        logger.warning(f"[safety] toxicity judge unavailable: {exc}")
        return None


_TRIAGE_SYS = (
    "You are a triage assistant for a telehealth platform. Given a patient's "
    "symptoms (and optional age/history), estimate clinical urgency. This is "
    "ADVISORY ONLY — a clinician makes the real decision, and a separate red-flag "
    "rule layer handles emergencies. Choose the single best level. Respond ONLY "
    'with JSON: {"urgency_level": '
    '"emergency|urgent|semi_urgent|non_urgent|self_care", "rationale": "<one sentence>"}.'
)


def judge_triage(symptoms: str, age: int | None, history: list[str] | None) -> Optional[TriageJudgement]:
    ctx = f"Symptoms: {symptoms}."
    if age is not None:
        ctx += f" Age: {age}."
    if history:
        ctx += f" History: {', '.join(history)}."
    try:
        return get_llm().generate_json(
            messages=[
                {"role": "system", "content": _TRIAGE_SYS},
                {"role": "user", "content": ctx},
            ],
            validate=TriageJudgement,
            max_tokens=200,
        )
    except LLMError as exc:
        logger.warning(f"[safety] triage judge unavailable: {exc}")
        return None
