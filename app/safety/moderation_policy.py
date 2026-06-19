"""
Content-moderation policy — maps raw signals to a graduated content-state decision.

Three actions, least → most restrictive:

  allow      → publish normally                       (visibility: public)
  quarantine → publish but limit reach + queue for     (visibility: limited)
               human review — the "quarantine" state
  block      → withhold from publishing; only for       (visibility: hidden)
               high-confidence violations

Principles for a health platform:
  * Escalate, don't delete. Default to the least restrictive action the evidence
    supports; borderline content is **quarantined for human review**, not removed.
  * Health misinformation is ADVISORY → quarantine + review, NEVER auto-block.
    False positives suppress *true* health info; the grounded RAG check and a human
    make the call.
  * Distress is a person-welfare signal, NOT a content violation: it never hides a
    post — it routes the user to the crisis/clinical workflow.
  * PII is redacted (safe_text) rather than blocking the post.

The result tells the backend exactly which content-state to apply. Every decision
is advisory — a human reviewer/moderator can always override.
"""

from __future__ import annotations

MODERATION_POLICY_VERSION = "moderation-policy/2026.06"

_ACTION_RANK = {"allow": 0, "quarantine": 1, "block": 2}
_VISIBILITY = {"allow": "public", "quarantine": "limited", "block": "hidden"}
_PRIO_RANK = {"none": 0, "low": 1, "normal": 2, "high": 3, "urgent": 4}


def decide(*, toxicity_label: str = "clean", toxicity_score: float = 0.0,
           toxic_keyword_hit: bool = False, health_claim: bool = False,
           pii_detected: bool = False, distress_escalate: bool = False) -> dict:
    """Combine moderation signals into a single content-state decision."""
    action = "allow"
    priority = "none"
    reasons: list[str] = []

    def _escalate_action(a: str) -> None:
        nonlocal action
        if _ACTION_RANK[a] > _ACTION_RANK[action]:
            action = a

    def _raise_priority(p: str) -> None:
        nonlocal priority
        if _PRIO_RANK[p] > _PRIO_RANK[priority]:
            priority = p

    # Toxicity — curated keyword hits are high-precision → block; an LLM-only
    # "maybe toxic" is quarantined for a human, not hard-blocked.
    if toxic_keyword_hit or (toxicity_label == "harassment" and toxicity_score >= 0.85):
        _escalate_action("block")
        _raise_priority("normal")
        reasons.append("High-confidence toxic/abusive content")
    elif toxicity_label not in ("clean", "") and toxicity_score >= 0.5:
        _escalate_action("quarantine")
        _raise_priority("high")
        reasons.append("Possible toxic content — limited pending review")

    # Health misinformation — advisory only, never auto-block.
    if health_claim:
        _escalate_action("quarantine")
        _raise_priority("high")
        reasons.append("Unverified health claim — limited pending misinfo/human review")

    # PII — redacted in place; doesn't change visibility on its own.
    if pii_detected:
        reasons.append("PII detected and redacted")

    # Distress — welfare escalation; never hides content, but tops the review queue.
    if distress_escalate:
        _raise_priority("urgent")
        reasons.append("Possible distress/self-harm — route user to crisis workflow")

    needs_review = action != "allow" or distress_escalate
    return {
        "action": action,
        "visibility": _VISIBILITY[action],
        "needs_human_review": needs_review,
        "review_priority": priority if needs_review else "none",
        "reasons": reasons,
        "policy_version": MODERATION_POLICY_VERSION,
    }
