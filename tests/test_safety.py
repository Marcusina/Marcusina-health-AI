"""
Tests for the Tier-1 safety lane (app/safety).

The deterministic rules run against the real config/*.json. The LLM judges are
monkeypatched so the decision logic — including the fail-safe paths — is tested
deterministically and without any network call or retry backoff.
"""

from __future__ import annotations

import pytest

from app.safety import assess_triage, assess_moderation, assess_distress
from app.safety.judges import DistressJudgement, ToxicityJudgement, TriageJudgement


# ── Triage ────────────────────────────────────────────────────────────────────

def test_triage_red_flag_is_emergency_without_llm(monkeypatch):
    # Red flag must short-circuit — the judge should never be consulted.
    monkeypatch.setattr("app.safety.judges.judge_triage",
                        lambda *a, **k: pytest.fail("LLM consulted despite red flag"))
    r = assess_triage("crushing chest pain radiating to the arm", age=60)
    assert r["urgency_level"] == "emergency"
    assert r["urgency_score"] == 1.0
    assert r["red_flag_symptoms"]
    assert r["llm_used"] is False


def test_triage_uses_llm_when_no_red_flag(monkeypatch):
    monkeypatch.setattr("app.safety.judges.judge_triage",
                        lambda *a, **k: TriageJudgement(urgency_level="urgent", rationale="x"))
    r = assess_triage("persistent moderate headache for three days")
    assert r["urgency_level"] == "urgent"
    assert r["llm_used"] is True


def test_triage_fails_safe_when_llm_down(monkeypatch):
    monkeypatch.setattr("app.safety.judges.judge_triage", lambda *a, **k: None)
    r = assess_triage("mild sore throat")
    assert r["urgency_level"] == "non_urgent"
    assert r["llm_used"] is False
    assert "unavailable" in r["reasoning"].lower()


def test_triage_skips_llm_when_not_requested(monkeypatch):
    monkeypatch.setattr("app.safety.judges.judge_triage",
                        lambda *a, **k: pytest.fail("LLM consulted with use_llm=False"))
    r = assess_triage("mild sore throat", use_llm=False)
    assert r["urgency_level"] == "non_urgent"


# ── Distress ──────────────────────────────────────────────────────────────────

def test_distress_no_pattern_is_clear():
    r = assess_distress("what a lovely sunny day, feeling great")
    assert r["detected"] is False
    assert r["escalate_to_human"] is False


def test_distress_high_severity_escalates(monkeypatch):
    monkeypatch.setattr("app.safety.judges.judge_distress",
                        lambda t: DistressJudgement(severity="high", escalate=True, rationale="x"))
    r = assess_distress("i want to end my life")
    assert r["detected"] is True
    assert r["escalate_to_human"] is True
    assert r["matched"]


def test_distress_benign_is_filtered_by_llm(monkeypatch):
    # Pattern fires but the LLM clears it as figurative → no escalation (FP filter).
    monkeypatch.setattr("app.safety.judges.judge_distress",
                        lambda t: DistressJudgement(severity="none", escalate=False, rationale="figurative"))
    r = assess_distress("ugh this deadline is killing me, i could just die of boredom")
    # a pattern may or may not match; if it did, LLM cleared it
    assert r["escalate_to_human"] is False
    assert r["detected"] is False


def test_distress_fails_safe_when_llm_down(monkeypatch):
    monkeypatch.setattr("app.safety.judges.judge_distress", lambda t: None)
    r = assess_distress("everyone would be better off without me")
    assert r["detected"] is True
    assert r["severity"] == "high"
    assert r["escalate_to_human"] is True


def test_distress_without_llm_still_escalates_on_match(monkeypatch):
    monkeypatch.setattr("app.safety.judges.judge_distress",
                        lambda t: pytest.fail("LLM consulted with use_llm=False"))
    r = assess_distress("everyone would be better off without me", use_llm=False)
    assert r["escalate_to_human"] is True


# ── Moderation ────────────────────────────────────────────────────────────────

def test_moderation_toxic_keyword_blocks(monkeypatch):
    # Keyword hit is authoritative; the toxicity judge must not be needed.
    monkeypatch.setattr("app.safety.judges.judge_toxicity",
                        lambda t: pytest.fail("LLM consulted despite keyword hit"))
    monkeypatch.setattr("app.safety.judges.judge_distress", lambda t: None)
    r = assess_moderation("you should kill yourself", context="comment")
    assert r["action"] == "block"
    assert r["toxicity"]["label"] == "toxic"
    assert r["toxicity"]["matched"]


def test_moderation_clean_text_allows():
    r = assess_moderation("Has anyone tried this new diabetes medication?", context="post")
    assert r["action"] == "allow"
    assert r["distress"]["detected"] is False


def test_moderation_deep_scan_uses_llm(monkeypatch):
    monkeypatch.setattr("app.safety.judges.judge_toxicity",
                        lambda t: ToxicityJudgement(label="harassment", action="block", rationale="x"))
    monkeypatch.setattr("app.safety.judges.judge_distress", lambda t: None)
    r = assess_moderation("subtle nasty insult without banned words", deep_scan=True)
    assert r["action"] == "block"
    assert r["llm_used"] is True


def test_moderation_surfaces_distress_separately_from_action(monkeypatch):
    # A distress message that isn't toxic: content is allowed, but distress escalates.
    monkeypatch.setattr("app.safety.judges.judge_distress",
                        lambda t: DistressJudgement(severity="high", escalate=True, rationale="x"))
    r = assess_moderation("i want to end my life", context="chat")
    assert r["action"] == "allow"
    assert r["distress"]["escalate_to_human"] is True
