"""
Tests for the content-moderation policy (app/safety/moderation_policy).

Locks in the graduated content-state semantics: allow / quarantine / block, the
"escalate, don't delete" default, and the health-platform rules (misinfo is never
auto-blocked; distress never hides content).
"""

from __future__ import annotations

from app.safety.moderation_policy import decide


def test_clean_content_is_allowed_public():
    d = decide()
    assert d["action"] == "allow"
    assert d["visibility"] == "public"
    assert d["needs_human_review"] is False
    assert d["review_priority"] == "none"


def test_toxic_keyword_blocks_and_hides():
    d = decide(toxic_keyword_hit=True)
    assert d["action"] == "block" and d["visibility"] == "hidden"
    assert d["needs_human_review"] is True


def test_high_score_harassment_blocks():
    assert decide(toxicity_label="harassment", toxicity_score=0.9)["action"] == "block"


def test_borderline_toxicity_quarantines_not_blocks():
    d = decide(toxicity_label="toxic", toxicity_score=0.6)
    assert d["action"] == "quarantine" and d["visibility"] == "limited"
    assert d["review_priority"] == "high"


def test_health_claim_quarantines_never_blocks():
    d = decide(health_claim=True)
    assert d["action"] == "quarantine"        # advisory — limited, not removed
    assert d["needs_human_review"] is True
    assert any("health claim" in r.lower() for r in d["reasons"])


def test_most_restrictive_signal_wins():
    # health claim (quarantine) + toxic keyword (block) → block, but both reasons kept
    d = decide(health_claim=True, toxic_keyword_hit=True)
    assert d["action"] == "block"
    assert len(d["reasons"]) >= 2


def test_pii_is_redacted_not_blocked():
    d = decide(pii_detected=True)
    assert d["action"] == "allow"             # redaction handles it; post still publishes
    assert d["needs_human_review"] is False
    assert any("pii" in r.lower() for r in d["reasons"])


def test_distress_escalates_without_hiding_content():
    d = decide(distress_escalate=True)
    assert d["action"] == "allow"             # never hide a post for distress
    assert d["needs_human_review"] is True
    assert d["review_priority"] == "urgent"
    assert any("distress" in r.lower() for r in d["reasons"])


def test_distress_with_misinfo_quarantines_and_urgent():
    d = decide(health_claim=True, distress_escalate=True)
    assert d["action"] == "quarantine"
    assert d["review_priority"] == "urgent"   # distress sets the priority ceiling
