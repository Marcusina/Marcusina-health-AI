"""
Support ticket assist: category + priority + summary + draft reply.

The draft is for a human agent — empathetic, professional, no medical advice, no
unkeepable promises. A deterministic distress check (app/safety/rules) runs
regardless of the LLM, forcing urgent priority + escalation if a self-harm
pattern fires.
"""

from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from app.llm import get_llm, LLMError
from app.safety import rules

SUPPORT_VERSION = "support-llm/2026.06"

_CATEGORIES = ("billing", "technical", "appointment", "prescription",
               "clinical_question", "account", "complaint", "feedback", "other")

_SYS = (
    "You are a support-desk assistant for a telehealth platform. From the support "
    "ticket, produce routing and a DRAFT reply for a HUMAN support agent to review "
    "and send. Rules:\n"
    "- The draft must be empathetic, professional, and concise; sign as 'the Marcusina "
    "Support team'.\n"
    "- NEVER give medical or clinical advice, diagnoses, medication, or dosage guidance. "
    "If the ticket is a medical/clinical question, set category 'clinical_question' and "
    "the draft must say it is being routed to a qualified clinician — do NOT answer the "
    "medical question.\n"
    "- Do not promise refunds, specific timelines, or outcomes; defer to the human agent.\n"
    "- If the ticket suggests self-harm or a crisis, set priority 'urgent'.\n"
    f"Category must be one of: {', '.join(_CATEGORIES)}.\n"
    'Respond ONLY with JSON: {"category": "...", "priority": "low|normal|high|urgent", '
    '"summary": "<one line>", "draft_reply": "<short reply>", "suggested_actions": ["..."]}.'
)


class SupportDraft(BaseModel):
    category: Literal["billing", "technical", "appointment", "prescription",
                      "clinical_question", "account", "complaint", "feedback", "other"]
    priority: Literal["low", "normal", "high", "urgent"]
    summary: str
    draft_reply: str
    suggested_actions: list[str] = Field(default_factory=list)


def draft_support_reply(subject: str, message: str,
                        category_hint: str | None = None) -> dict:
    """Draft a support reply + routing. Always returns (degrades if LLM is down)."""
    ticket = f"Subject: {subject}\n\nMessage: {message}"
    if category_hint:
        ticket += f"\n\n(Customer-selected category hint: {category_hint}.)"

    # Deterministic distress net — runs regardless of the LLM.
    distress_hits = rules.match_distress(f"{subject} {message}")

    try:
        d: SupportDraft = get_llm().generate_json(
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": ticket}],
            validate=SupportDraft, max_tokens=500,
        )
        out = {
            "category": d.category, "priority": d.priority, "summary": d.summary,
            "draft_reply": d.draft_reply, "suggested_actions": list(d.suggested_actions),
            "llm_used": True, "degraded": False,
        }
    except LLMError as exc:
        logger.warning(f"[support] assist LLM unavailable, degraded: {exc}")
        out = {
            "category": category_hint if category_hint in _CATEGORIES else "other",
            "priority": "normal", "summary": "",
            "draft_reply": "", "suggested_actions": [],
            "llm_used": False, "degraded": True,
        }

    # Distress overrides priority and forces escalation (recall over precision).
    out["distress_flag"] = bool(distress_hits)
    if distress_hits:
        out["priority"] = "urgent"
        out["suggested_actions"] = (
            ["Escalate: possible distress/self-harm — route to crisis/clinical workflow."]
            + out["suggested_actions"]
        )

    out["needs_human_review"] = True   # always a draft — a human sends it
    out["model_version"] = SUPPORT_VERSION
    return out
