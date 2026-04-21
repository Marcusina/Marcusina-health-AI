"""
app/utils/config_loader.py
===========================
Loads all runtime configuration from JSON files in the /config directory.
Nothing is hardcoded in application code.

To add a new red-flag symptom: edit config/red_flags.json
To add a new ICD code: edit config/icd_map.json
To add a new toxic phrase: edit config/toxic_keywords.json

"""

from __future__ import annotations
import json
import re
from pathlib import Path
from functools import lru_cache
from loguru import logger

CONFIG_DIR = Path("config")


def _bound_pattern(p: str) -> str:
    """
    Add \\b word boundaries to each top-level alternative in a pattern.
    Splits on | that are NOT inside parentheses, so grouped patterns
    like (so much|too much) are left intact.

    Example: "die|just let me die"  →  "\\bdie\\b|\\bjust let me die\\b"
    Example: "blood|bleed|bleeding" →  "\\bblood\\b|\\bbleed\\b|\\bbleeding\\b"
    """
    alts: list[str] = []
    current: list[str] = []
    depth = 0

    for ch in p:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '|' and depth == 0:
            alts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    alts.append(''.join(current))

    bounded: list[str] = []
    for alt in alts:
        stripped = alt.strip()
        if not stripped:
            bounded.append(stripped)
            continue
        pre = r"\b" if stripped[0].isalnum() or stripped[0] == "'" else ""
        suf = r"\b" if stripped[-1].isalnum() or stripped[-1] == "'" else ""
        bounded.append(f"{pre}{stripped}{suf}")

    return "|".join(bounded)


def _load_json(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Create it — see README or the config/ folder instructions."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def get_red_flags() -> list[str]:
    """Emergency symptom keywords. Loaded once per process, cached."""
    data = _load_json("red_flags.json")
    logger.info(f"Loaded {len(data['keywords'])} red-flag keywords from config.")
    return data["keywords"]


@lru_cache(maxsize=None)
def get_health_claim_pattern() -> re.Pattern:
    """Compiled regex for health claim detection."""
    patterns = _load_json("health_claim_patterns.json")["patterns"]
    combined = "|".join(f"(?:{_bound_pattern(p)})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


@lru_cache(maxsize=None)
def get_distress_pattern() -> re.Pattern:
    """Compiled regex for mental health distress detection."""
    patterns = _load_json("distress_patterns.json")["patterns"]
    combined = "|".join(f"(?:{_bound_pattern(p)})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


@lru_cache(maxsize=None)
def get_toxic_keywords() -> frozenset[str]:
    """Frozen set of toxic keyword phrases."""
    return frozenset(_load_json("toxic_keywords.json")["keywords"])


@lru_cache(maxsize=None)
def get_icd_map() -> dict[str, str]:
    """Keyword to ICD-10 code mapping."""
    return _load_json("icd_map.json")["mappings"]


@lru_cache(maxsize=None)
def get_specialty_map() -> dict[str, str]:
    """Symptom keyword to medical specialty."""
    return _load_json("specialty_map.json")["mappings"]


def reload_all() -> None:
    """
    Clear all cached configs and reload from disk.
    Call this after editing any config JSON without restarting.
    """
    get_red_flags.cache_clear()
    get_health_claim_pattern.cache_clear()
    get_distress_pattern.cache_clear()
    get_toxic_keywords.cache_clear()
    get_icd_map.cache_clear()
    get_specialty_map.cache_clear()
    logger.info("All config caches cleared — will reload from disk on next request.")