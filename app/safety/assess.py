"""
Decision logic — combines the deterministic rules (rules.py) with optional LLM
escalation (judges.py) into the responses the sync endpoints return.

Safety invariants:
  * Triage red-flag keywords are authoritative → emergency, always.
  * A distress pattern firing always routes to a human UNLESS the LLM positively
    clears it as benign. If the LLM can't be reached, we escalate (recall > precision).
  * Toxicity is precision-oriented: curated toxic keywords block; otherwise we
    only act on a positive LLM judgement, and default to allow when uncertain.
"""

from __future__ import annotations

from app.safety import rules
from app.safety import judges

TRIAGE_VERSION = "triage-rules+llm/2026.06"
MODERATION_VERSION = "moderation-rules+llm/2026.06"

_URGENCY_SCORE = {
    "emergency": 0.95, "urgent": 0.80, "semi_urgent": 0.60,
    "non_urgent": 0.40, "self_care": 0.20,
}
_TOXICITY_SCORE = {"clean": 0.02, "toxic": 0.85, "harassment": 0.90}

_SELF_CARE_ADVICE = ("Rest, stay hydrated, and monitor your symptoms. "
                     "Seek care promptly if anything worsens or new symptoms appear.")


# ── Triage ────────────────────────────────────────────────────────────────────

def assess_triage(symptoms: str, age: int | None = None,
                  medical_history: list[str] | None = None,
                  use_llm: bool = True) -> dict:
    red = rules.match_red_flags(symptoms)
    specialty = rules.route_specialty(symptoms)

    # 1. Red flags are authoritative — emergency, no LLM needed.
    if red:
        return {
            "urgency_level": "emergency", "urgency_score": 1.0,
            "red_flag_symptoms": red, "recommended_specialty": specialty,
            "reasoning": f"Red-flag symptom(s) detected: {', '.join(red)}. "
                         f"Route to emergency / 24-7 clinical pathway.",
            "self_care_advice": None, "llm_used": False,
            "model_version": TRIAGE_VERSION,
        }

    # 2. No red flags — ask the LLM for an advisory urgency opinion.
    if use_llm:
        j = judges.judge_triage(symptoms, age, medical_history)
        if j is not None:
            level = j.urgency_level
            return {
                "urgency_level": level, "urgency_score": _URGENCY_SCORE[level],
                "red_flag_symptoms": [], "recommended_specialty": specialty,
                "reasoning": f"Advisory (LLM): {j.rationale}",
                "self_care_advice": _SELF_CARE_ADVICE if level == "self_care" else None,
                "llm_used": True, "model_version": TRIAGE_VERSION,
            }
        # LLM unavailable — conservative default; the red-flag net already ran.
        return {
            "urgency_level": "non_urgent", "urgency_score": 0.40,
            "red_flag_symptoms": [], "recommended_specialty": specialty,
            "reasoning": "No red flags detected. Advisory model unavailable — "
                         "clinician review advised.",
            "self_care_advice": None, "llm_used": False,
            "model_version": TRIAGE_VERSION,
        }

    # 3. LLM not requested.
    return {
        "urgency_level": "non_urgent", "urgency_score": 0.40,
        "red_flag_symptoms": [], "recommended_specialty": specialty,
        "reasoning": "No red flags detected. Urgency model not requested.",
        "self_care_advice": None, "llm_used": False,
        "model_version": TRIAGE_VERSION,
    }


# ── Distress ──────────────────────────────────────────────────────────────────

def assess_distress(text: str, use_llm: bool = True) -> dict:
    matched = rules.match_distress(text)

    # No pattern fired — trust the high-recall net.
    if not matched:
        return {"detected": False, "severity": "none",
                "escalate_to_human": False, "matched": [], "llm_used": False}

    # Pattern fired — disambiguate with the LLM if we can.
    if use_llm:
        j = judges.judge_distress(text)
        if j is not None:
            detected = j.severity != "none"
            return {
                "detected": detected, "severity": j.severity,
                "escalate_to_human": bool(j.escalate or j.severity == "high"),
                "matched": matched, "llm_used": True,
            }
        # LLM down — fail safe: a distress pattern fired and we can't clear it.
        return {"detected": True, "severity": "high",
                "escalate_to_human": True, "matched": matched, "llm_used": False}

    # LLM not requested — still fail safe on a fired pattern.
    return {"detected": True, "severity": "low",
            "escalate_to_human": True, "matched": matched, "llm_used": False}


# ── Moderation (toxicity + distress in one call) ──────────────────────────────

def assess_moderation(text: str, context: str = "post", deep_scan: bool = False) -> dict:
    toxic = rules.match_toxic(text)
    # Distress is safety-critical and patterns are rare — always allow LLM disambiguation.
    distress = assess_distress(text, use_llm=True)

    tox_llm_used = False
    if toxic:
        # Curated harmful phrases → remove (per moderation policy).
        toxicity = {"score": 0.95, "label": "toxic", "matched": toxic}
        action = "block"
    elif deep_scan:
        j = judges.judge_toxicity(text)
        if j is not None:
            tox_llm_used = True
            toxicity = {"score": _TOXICITY_SCORE[j.label], "label": j.label, "matched": []}
            action = j.action
        else:
            # No keyword hit, LLM down — precision-oriented default: allow.
            toxicity = {"score": 0.0, "label": "clean", "matched": []}
            action = "allow"
    else:
        toxicity = {"score": 0.0, "label": "clean", "matched": []}
        action = "allow"

    return {
        "action": action,
        "toxicity": toxicity,
        "distress": {k: distress[k] for k in ("detected", "severity", "escalate_to_human", "matched")},
        "llm_used": bool(tox_llm_used or distress["llm_used"]),
        "model_version": MODERATION_VERSION,
    }
