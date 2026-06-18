"""
Tests for support-desk assist (app/support).

The LLM is stubbed. The deterministic distress net runs against the real config,
so the self-harm override is exercised without a model server.
"""

from __future__ import annotations

import pytest

from app.support.assist import draft_support_reply, SupportDraft
from app.llm import LLMUnavailable


class FakeLLM:
    def __init__(self, result=None, raise_exc=None):
        self.result, self.raise_exc = result, raise_exc

    def generate_json(self, **kwargs):
        if self.raise_exc:
            raise self.raise_exc
        return self.result


def _draft(**over):
    base = dict(category="billing", priority="normal", summary="Billing question",
                draft_reply="Hi, thanks for reaching out...", suggested_actions=["check invoice"])
    base.update(over)
    return SupportDraft(**base)


def test_support_assist_success(monkeypatch):
    monkeypatch.setattr("app.support.assist.get_llm", lambda: FakeLLM(result=_draft()))
    out = draft_support_reply("Charged twice", "I was billed twice this month.")
    assert out["category"] == "billing"
    assert out["draft_reply"]
    assert out["needs_human_review"] is True     # always a draft
    assert out["distress_flag"] is False
    assert out["llm_used"] is True


def test_support_assist_clinical_question_routed(monkeypatch):
    # The model classifies a medical question as clinical_question (no advice given).
    monkeypatch.setattr("app.support.assist.get_llm",
        lambda: FakeLLM(result=_draft(category="clinical_question",
                                      draft_reply="We're routing your question to a clinician.")))
    out = draft_support_reply("Question", "What dose of my medication should I take?")
    assert out["category"] == "clinical_question"
    assert "clinician" in out["draft_reply"].lower()


def test_support_assist_distress_forces_urgent(monkeypatch):
    # Even with a calm LLM draft, a self-harm pattern forces urgent + escalation.
    monkeypatch.setattr("app.support.assist.get_llm",
        lambda: FakeLLM(result=_draft(category="other", priority="low")))
    out = draft_support_reply("help", "i want to end my life, nothing helps anymore")
    assert out["distress_flag"] is True
    assert out["priority"] == "urgent"
    assert any("Escalate" in a for a in out["suggested_actions"])


def test_support_assist_distress_caught_even_when_llm_down(monkeypatch):
    monkeypatch.setattr("app.support.assist.get_llm",
                        lambda: FakeLLM(raise_exc=LLMUnavailable("gpu down")))
    out = draft_support_reply("help", "everyone would be better off without me")
    assert out["degraded"] is True
    assert out["llm_used"] is False
    assert out["distress_flag"] is True
    assert out["priority"] == "urgent"


def test_support_assist_degraded_keeps_category_hint(monkeypatch):
    monkeypatch.setattr("app.support.assist.get_llm",
                        lambda: FakeLLM(raise_exc=LLMUnavailable("down")))
    out = draft_support_reply("Refund", "I want a refund please", category_hint="billing")
    assert out["degraded"] is True
    assert out["category"] == "billing"
    assert out["needs_human_review"] is True
