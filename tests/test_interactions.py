"""
Tests for the drug-drug interaction checker (app/meds).

The curated rule base is the authoritative safety layer, so these assert the
deterministic behaviour directly — no network, no LLM (the advisory pass is
exercised with a stub).
"""

from __future__ import annotations

from app.meds import check_interactions
from app.meds.normalize import normalize_name, canonical_tokens


# ── normalization ─────────────────────────────────────────────────────────────

def test_brand_maps_to_generic():
    assert normalize_name("Coumadin") == "warfarin"
    assert normalize_name("Advil 200mg") == "ibuprofen"
    assert normalize_name("Viagra 50 mg") == "sildenafil"


def test_salt_and_form_suffixes_resolve():
    assert normalize_name("isosorbide mononitrate") == "isosorbide"
    assert normalize_name("metformin hydrochloride") == "metformin"
    assert normalize_name("sertraline HCl 50mg") == "sertraline"


def test_class_tokens_include_generic_and_class():
    generic, tokens = canonical_tokens("ibuprofen")
    assert generic == "ibuprofen"
    assert "nsaid" in tokens


# ── deterministic matching ────────────────────────────────────────────────────

def test_warfarin_plus_nsaid_is_major():
    r = check_interactions(["Coumadin 5mg", "Advil"])
    assert r["interaction_count"] == 1
    i = r["interactions"][0]
    assert {i["drug_a"], i["drug_b"]} == {"warfarin", "ibuprofen"}
    assert i["severity"] == "major"


def test_sildenafil_plus_nitrate_is_contraindicated():
    r = check_interactions(["Viagra", "isosorbide mononitrate"])
    assert r["has_contraindication"] is True
    assert r["highest_severity"] == "contraindicated"


def test_class_level_rule_matches_any_member():
    # lisinopril is an ACE inhibitor; rule is "ace_inhibitor + spironolactone"
    r = check_interactions(["lisinopril", "spironolactone"])
    assert r["interaction_count"] == 1
    assert r["interactions"][0]["severity"] == "major"


def test_duplicate_pair_collapses_to_most_severe():
    # spironolactone matches both its name and its potassium-sparing-diuretic class
    # for the ACE-inhibitor rule — must yield ONE finding for that pair, not two.
    r = check_interactions(["lisinopril", "spironolactone", "potassium chloride"])
    pairs = {frozenset((i["drug_a"], i["drug_b"])) for i in r["interactions"]}
    assert len(pairs) == r["interaction_count"]            # no duplicate pairs
    assert frozenset(("lisinopril", "spironolactone")) in pairs


def test_safe_pair_is_not_flagged():
    r = check_interactions(["metformin", "atorvastatin"])
    assert r["interaction_count"] == 0
    assert r["highest_severity"] == "none"


# ── safety invariants ─────────────────────────────────────────────────────────

def test_no_known_interaction_is_not_a_clearance():
    r = check_interactions(["acetaminophen", "vitamin c"])
    assert r["interaction_count"] == 0
    assert "vitamin c" in r["unrecognized"]               # unknown drug surfaced, not hidden
    assert r["needs_human_review"] is True
    assert "does NOT mean" in r["advisory"]


def test_always_needs_human_review():
    assert check_interactions(["warfarin", "aspirin"])["needs_human_review"] is True


def test_single_med_has_no_pairs():
    r = check_interactions(["warfarin"])
    assert r["interaction_count"] == 0
    assert r["medications_checked"] == ["warfarin"]


# ── optional LLM advisory pass (stubbed; additive only) ───────────────────────

def test_llm_advisory_is_additive_and_labelled(monkeypatch):
    import app.meds.interactions as mod

    class _FakeLLM:
        def generate_json(self, *, messages, validate, max_tokens):
            return validate(additional=[{
                "drug_a": "drugx", "drug_b": "drugy",
                "severity": "moderate", "effect": "stub candidate"}])

    monkeypatch.setattr(mod, "get_llm", lambda: _FakeLLM())
    r = check_interactions(["warfarin", "aspirin"], use_llm=True)
    assert r["llm_used"] is True
    assert r["llm_advisory"] and r["llm_advisory"][0]["source"] == "llm-advisory-unverified"
    # the deterministic finding still stands on its own
    assert r["interaction_count"] == 1


def test_llm_failure_leaves_deterministic_result(monkeypatch):
    import app.meds.interactions as mod
    from app.llm import LLMError

    class _DownLLM:
        def generate_json(self, **_):
            raise LLMError("gpu asleep")

    monkeypatch.setattr(mod, "get_llm", lambda: _DownLLM())
    r = check_interactions(["warfarin", "aspirin"], use_llm=True)
    assert r["llm_used"] is False
    assert r["llm_advisory"] == []
    assert r["interaction_count"] == 1        # safety result unaffected
