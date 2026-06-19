"""
Medication safety lane — deterministic drug-drug interaction checking with an
optional advisory LLM pass. The curated rule base is authoritative; the LLM can
only add candidates for human review, never clear a combination.

Public surface:
    check_interactions(medications, use_llm=False) -> dict matching DrugInteractionResult
"""

from app.meds.interactions import check_interactions, MEDS_VERSION

__all__ = ["check_interactions", "MEDS_VERSION"]
