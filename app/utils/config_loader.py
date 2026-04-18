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
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.IGNORECASE)


@lru_cache(maxsize=None)
def get_distress_pattern() -> re.Pattern:
    """Compiled regex for mental health distress detection."""
    patterns = _load_json("distress_patterns.json")["patterns"]
    combined = "|".join(f"(?:{p})" for p in patterns)
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