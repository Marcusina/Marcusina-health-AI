"""
Tests for the clinical generation lane (app/clinical) — SOAP + summary.

The LLM is stubbed (a fake client whose generate_json returns a validated model
or raises), so the orchestration, ICD grounding, and degraded-fallback paths are
tested without a model server.
"""

from __future__ import annotations

import pytest

from app.clinical.soap import generate_soap, suggest_icd, SoapNote
from app.clinical.summary import generate_summary, VisitSummary
from app.llm import LLMUnavailable


class FakeLLM:
    def __init__(self, result=None, raise_exc=None):
        self.result = result
        self.raise_exc = raise_exc

    def generate_json(self, **kwargs):
        if self.raise_exc:
            raise self.raise_exc
        return self.result


# ── ICD grounding ─────────────────────────────────────────────────────────────

def test_soapnote_coerces_varied_entity_shapes():
    # Models return entity fields as strings / dicts / mixed lists — all must coerce.
    note = SoapNote(
        subjective="s", objective="o", assessment="a", plan="p",
        vitals="Temperature: 37.9C",                       # bare string
        medications=[{"name": "paracetamol"}],             # list of dict
        symptoms=["cough", {"finding": "fever"}],          # mixed list
        diagnoses=None, procedures=[],
    )
    assert note.vitals == ["Temperature: 37.9C"]
    assert note.medications == ["name: paracetamol"]
    assert note.symptoms == ["cough", "finding: fever"]
    assert note.diagnoses == []


def test_suggest_icd_maps_known_terms():
    codes = suggest_icd(["Type 2 diabetes mellitus", "hypertension"], [])
    assert "E11" in codes      # diabetes
    assert "I10" in codes      # hypertension


def test_suggest_icd_is_bounded_and_deduped():
    codes = suggest_icd(["diabetes", "diabetes mellitus", "type 2 diabetes"], [], limit=5)
    assert codes == ["E11"]    # all map to the same code → deduped


def test_suggest_icd_skips_negated_findings():
    positive = suggest_icd([], ["chest pain"])
    assert "R07.9" in positive                              # positive maps
    assert suggest_icd([], ["No chest pain"]) == []          # negated → no code
    assert suggest_icd([], ["denies shortness of breath"]) == []
    # mixed: a real cough still maps, but the negated chest pain must not leak in
    mixed = suggest_icd([], ["cough", "no chest pain", "no shortness of breath"])
    assert "R05" in mixed                                   # cough mapped
    assert "R07.9" not in mixed and "R06.02" not in mixed   # negatives skipped


# ── SOAP ──────────────────────────────────────────────────────────────────────

def test_generate_soap_success(monkeypatch):
    note = SoapNote(
        subjective="Patient reports a cough for 3 days.",
        objective="Temp 37.8C. Chest clear.",
        assessment="Likely viral upper respiratory infection.",
        plan="Rest, fluids, paracetamol. Review if worse.",
        medications=["paracetamol"], diagnoses=["viral upper respiratory infection"],
        symptoms=["cough"], procedures=[], vitals=["Temp 37.8C"],
    )
    monkeypatch.setattr("app.clinical.soap.get_llm", lambda: FakeLLM(result=note))
    out = generate_soap("Doctor: ... Patient: I've had a cough ...", patient_id="p1")
    assert out["llm_used"] is True
    assert out["degraded"] is False
    assert set(out["soap_note"]) == {"subjective", "objective", "assessment", "plan"}
    assert out["extracted_entities"]["medications"] == ["paracetamol"]
    assert isinstance(out["icd_suggestions"], list)


def test_generate_soap_degrades_when_llm_down(monkeypatch):
    monkeypatch.setattr("app.clinical.soap.get_llm",
                        lambda: FakeLLM(raise_exc=LLMUnavailable("gpu down")))
    out = generate_soap("Doctor: ... Patient: chest discomfort ...")
    assert out["degraded"] is True
    assert out["llm_used"] is False
    # still returns a usable (clearly-marked) structure for the clinician
    assert "AUTO-DRAFT" in out["soap_note"]["subjective"]
    assert set(out["soap_note"]) == {"subjective", "objective", "assessment", "plan"}


# ── Summary ───────────────────────────────────────────────────────────────────

def test_generate_summary_success(monkeypatch):
    s = VisitSummary(
        summary="You came in with a cough. It's likely a mild viral infection.",
        next_steps=["Rest and drink fluids", "Take paracetamol if needed"],
        when_to_seek_help=["Trouble breathing", "Fever above 39C for more than 3 days"],
    )
    monkeypatch.setattr("app.clinical.summary.get_llm", lambda: FakeLLM(result=s))
    out = generate_summary("Doctor: ... Patient: cough ...", session_id="s1")
    assert out["llm_used"] is True
    assert out["degraded"] is False
    assert out["next_steps"]
    assert out["disclaimer"]            # always present


def test_generate_summary_degrades_when_llm_down(monkeypatch):
    monkeypatch.setattr("app.clinical.summary.get_llm",
                        lambda: FakeLLM(raise_exc=LLMUnavailable("gpu down")))
    out = generate_summary("Doctor: ... Patient: ...")
    assert out["degraded"] is True
    assert out["llm_used"] is False
    assert out["disclaimer"]
