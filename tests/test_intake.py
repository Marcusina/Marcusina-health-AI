"""
Tests for the pre-consultation symptom intake assistant (app/intake).

Both the triage net and the LLM are stubbed so these run offline and assert the
intake's own logic: triage is authoritative for urgency, the LLM only structures,
and everything degrades safely.
"""

from __future__ import annotations

import app.intake.assistant as mod
from app.intake import build_intake
from app.llm import LLMError


def _triage(urgency="non_urgent", red=None, specialty="General Practitioner"):
    return {
        "urgency_level": urgency,
        "red_flag_symptoms": red or [],
        "recommended_specialty": specialty,
        "urgency_score": 0.5, "reasoning": "stub", "llm_used": False,
        "model_version": "stub",
    }


class _FakeLLM:
    def __init__(self, intake=None, raise_=None):
        self._intake = intake
        self._raise = raise_

    def generate_json(self, *, messages, validate, max_tokens):
        if self._raise:
            raise self._raise
        return validate(**self._intake)


def test_llm_structures_intake_when_available(monkeypatch):
    monkeypatch.setattr(mod, "assess_triage", lambda *a, **k: _triage())
    monkeypatch.setattr(mod, "get_llm", lambda: _FakeLLM(intake={
        "chief_complaint": "headache",
        "structured_summary": "Throbbing headache for 2 days.",
        "clarifying_questions": ["Any fever?", "Any vision changes?"]}))

    r = build_intake("My head has been pounding for two days", age=30)
    assert r["chief_complaint"] == "headache"
    assert r["degraded"] is False and r["llm_used"] is True
    assert r["emergency"] is False
    assert "review this before your consultation" in r["patient_guidance"]
    assert r["needs_human_review"] is True


def test_triage_red_flag_is_authoritative_and_not_downgraded(monkeypatch):
    # Triage says emergency; the LLM intake carries no urgency — emergency must stand.
    monkeypatch.setattr(mod, "assess_triage",
                        lambda *a, **k: _triage(urgency="emergency", red=["chest pain"]))
    monkeypatch.setattr(mod, "get_llm", lambda: _FakeLLM(intake={
        "chief_complaint": "chest discomfort",
        "structured_summary": "Chest pain radiating to the left arm.",
        "clarifying_questions": ["When did it start?"]}))

    r = build_intake("crushing chest pain spreading to my left arm", age=58)
    assert r["emergency"] is True
    assert r["urgency_level"] == "emergency"
    assert r["red_flag_symptoms"] == ["chest pain"]
    assert "immediate medical care" in r["patient_guidance"]


def test_degrades_safely_when_llm_down(monkeypatch):
    monkeypatch.setattr(mod, "assess_triage", lambda *a, **k: _triage())
    monkeypatch.setattr(mod, "get_llm", lambda: _FakeLLM(raise_=LLMError("gpu asleep")))

    r = build_intake("sore throat and a mild cough", age=25)
    assert r["degraded"] is True and r["llm_used"] is False
    assert r["clarifying_questions"]                 # skeleton still gives questions
    assert r["structured_summary"] == "sore throat and a mild cough"
    assert r["needs_human_review"] is True


def test_use_llm_false_skips_llm(monkeypatch):
    monkeypatch.setattr(mod, "assess_triage", lambda *a, **k: _triage())
    # get_llm must never be called; make it explode if it is
    monkeypatch.setattr(mod, "get_llm", lambda: (_ for _ in ()).throw(AssertionError("called")))

    r = build_intake("mild headache", use_llm=False)
    assert r["degraded"] is True and r["llm_used"] is False
