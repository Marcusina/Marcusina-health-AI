"""
Tier-1 SYNCHRONOUS safety endpoints.

Unlike the async (Celery) endpoints in v1.py, these run inference inline and
return the result in the same response — the backend calls them to **gate a user
action** (deliver a message? route an appointment as emergency?).

They are intentionally plain `def` handlers: FastAPI runs them in a threadpool, so
the blocking rules + (optional) LLM call never block the event loop. The rules
layer is GPU-independent and instant; only ambiguous cases touch the LLM, which
fails safe if the on-demand GPU is down.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from loguru import logger

from app.core.security import verify_internal_secret
from app.models.schemas import (
    TriageSyncRequest, TriageResult,
    ModerateTextRequest, ModerateResult,
)
from app.safety import assess_triage, assess_moderation

router = APIRouter(dependencies=[Depends(verify_internal_secret)], tags=["Tier-1 sync"])


@router.post("/triage", response_model=TriageResult,
             summary="Synchronous triage — urgency + specialty (rules + LLM advisory)")
def triage(req: TriageSyncRequest) -> TriageResult:
    result = assess_triage(
        symptoms=req.symptoms, age=req.age,
        medical_history=req.medical_history, use_llm=req.use_llm,
    )
    if result["red_flag_symptoms"]:
        logger.warning(f"[triage] RED FLAG for patient={req.patient_id}: "
                       f"{result['red_flag_symptoms']}")
    return TriageResult(**result)


@router.post("/moderate/text", response_model=ModerateResult,
             summary="Synchronous moderation — toxicity + distress in one call")
def moderate_text(req: ModerateTextRequest) -> ModerateResult:
    result = assess_moderation(text=req.text, context=req.context, deep_scan=req.deep_scan)
    if result["distress"]["escalate_to_human"]:
        logger.warning(f"[moderate] DISTRESS escalation for entity={req.entity_id} "
                       f"(severity={result['distress']['severity']})")
    return ModerateResult(**result)
