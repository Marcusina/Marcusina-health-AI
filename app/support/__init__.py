"""
Support-desk assist (local LLM).

Drafts a reply and routes a support ticket — a DRAFT for a human agent to review,
edit, and send. It never auto-sends, and it never gives medical advice (clinical
questions are routed to a clinician, not answered). The deterministic distress net
also runs, so self-harm content in a ticket is flagged for escalation even if the
LLM is unavailable.
"""

from app.support.assist import draft_support_reply

__all__ = ["draft_support_reply"]
