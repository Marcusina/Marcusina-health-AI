"""
Pre-consultation symptom intake assistant — structures a patient's complaint for
the clinician and proposes clarifying questions. Advisory only; the deterministic
red-flag triage net is authoritative for urgency and is never downgraded.

Public surface:
    build_intake(symptoms, ...) -> dict matching SymptomIntakeResult
"""

from app.intake.assistant import build_intake, INTAKE_VERSION

__all__ = ["build_intake", "INTAKE_VERSION"]
