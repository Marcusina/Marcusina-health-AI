"""
Deterministic rule matchers over config/*.json.

Pure functions, no models, no network — the always-available safety net. All are
case-insensitive. These are tuned for **recall**: they catch broadly and let the
LLM layer (judges.py) refine. Loaded once and cached by config_loader.
"""

from __future__ import annotations

from app.utils.config_loader import (
    get_red_flags,
    get_distress_pattern,
    get_toxic_keywords,
    get_specialty_map,
)


def match_red_flags(text: str) -> list[str]:
    """Emergency red-flag keywords present in the text (substring, case-insensitive)."""
    t = text.lower()
    return [kw for kw in get_red_flags() if kw in t]


def match_distress(text: str) -> list[str]:
    """Distinct distress/ideation phrases the regex matched (deduped, order-stable)."""
    seen: dict[str, None] = {}
    for m in get_distress_pattern().finditer(text):
        phrase = m.group(0).strip().lower()
        if phrase:
            seen.setdefault(phrase, None)
    return list(seen)


def match_toxic(text: str) -> list[str]:
    """Curated toxic/harmful phrases present in the text (substring, case-insensitive)."""
    t = text.lower()
    return [kw for kw in get_toxic_keywords() if kw in t]


def route_specialty(text: str, default: str = "General Practitioner") -> str:
    """First specialty whose keyword appears in the text; else the default."""
    t = text.lower()
    for keyword, specialty in get_specialty_map().items():
        if keyword in t:
            return specialty
    return default
