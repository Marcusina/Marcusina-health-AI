"""
Tier-1 safety lane — synchronous, deterministic-first inference for the calls
that gate a user action: triage, distress detection, content moderation.

Architecture (why this module exists separately from the Celery async tasks):

  1. A **deterministic rules layer** (red-flag keywords, distress regex, toxic
     keywords from config/*.json) is the authoritative safety net. It is instant,
     pure-Python, never depends on a GPU, and is tuned for **recall** — it would
     rather over-flag than miss.
  2. The **local LLM escalates ambiguous cases only** (e.g. a distress pattern
     fired — is it real ideation or "I'm done for today"?). It refines severity
     and filters false positives.
  3. **Fail safe.** If the LLM is unavailable (the on-demand GPU is spun down),
     safety-critical decisions over-escalate rather than silently downgrade.

Public surface:
    assess_triage(...)     -> dict matching TriageResult
    assess_moderation(...) -> dict matching ModerateResult (toxicity + distress)
    assess_distress(...)   -> dict matching DistressSignal
"""

from app.safety.assess import assess_triage, assess_moderation, assess_distress

__all__ = ["assess_triage", "assess_moderation", "assess_distress"]
