"""
Tier-3 clinical-assist endpoints (synchronous, advisory).

These support a clinician/patient workflow inline rather than gating a user action
like the Tier-1 safety lane. Everything here is **advisory and human-confirmed**:

  * POST /api/v1/medications/interactions — drug-drug interaction check (#12).
    Deterministic curated rules are authoritative; an optional LLM pass only adds
    candidates for review. Never declares a combination safe.
  * POST /api/v1/intake/symptoms — pre-consultation symptom intake (#13). The
    red-flag triage net is authoritative for urgency; the LLM only structures the
    complaint and proposes clarifying questions. Advisory, never a diagnosis.

Plain `def` handlers — FastAPI threadpools them so the (optional) LLM call and the
rule lookups don't block the event loop.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from loguru import logger

from app.core.security import verify_internal_secret
from app.models.schemas import (
    DrugInteractionRequest, DrugInteractionResult,
    SymptomIntakeRequest, SymptomIntakeResult,
)
from app.meds import check_interactions
from app.intake import build_intake

router = APIRouter(dependencies=[Depends(verify_internal_secret)], tags=["Tier-3 clinical-assist"])


@router.post("/medications/interactions", response_model=DrugInteractionResult,
             summary="Drug-drug interaction check (curated rules + optional LLM advisory)")
def medication_interactions(req: DrugInteractionRequest) -> DrugInteractionResult:
    result = check_interactions(req.medications, use_llm=req.use_llm)
    if result["has_contraindication"]:
        pairs = [(i["drug_a"], i["drug_b"]) for i in result["interactions"]
                 if i["severity"] == "contraindicated"]
        logger.warning(f"[meds] CONTRAINDICATION for patient={req.patient_id}: {pairs}")
    return DrugInteractionResult(**result)


@router.post("/intake/symptoms", response_model=SymptomIntakeResult,
             summary="Pre-consultation symptom intake (triage net authoritative; LLM structures)")
def symptom_intake(req: SymptomIntakeRequest) -> SymptomIntakeResult:
    result = build_intake(
        req.symptoms, age=req.age, sex=req.sex, duration=req.duration,
        existing_conditions=req.existing_conditions, medications=req.medications,
        use_llm=req.use_llm,
    )
    if result["emergency"]:
        logger.warning(f"[intake] EMERGENCY red-flags for patient={req.patient_id}: "
                       f"{result['red_flag_symptoms']}")
    return SymptomIntakeResult(**result)
