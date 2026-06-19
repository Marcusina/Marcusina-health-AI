"""
Drug-drug interaction check (advisory, clinician-confirmed).

Safety model — mirrors the triage/RAG lanes:

  * A **curated, deterministic rule base** (`data/known_interactions.jsonl`) is the
    authoritative layer. Pairs of medications are matched on canonical tokens
    (generic name OR drug class), so "NSAID + warfarin" catches every NSAID.
  * The **LLM is advisory-only and strictly additive** (opt-in): it may surface
    *additional candidate* interactions for a clinician to verify, but it can never
    clear a combination or downgrade a curated finding. Hallucinated reassurance is
    the dangerous failure mode, so the LLM is never allowed to say "safe".
  * **Absence of a flag is never "safe".** When no known interaction is found, the
    result says exactly that — limited reference set, not a clearance — and any
    medication we couldn't recognize is reported as unrecognized, not cleared.

Every result is advisory and `needs_human_review=True`: a clinician/pharmacist
confirms against a complete interaction database before any action.
"""

from __future__ import annotations

import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Optional

from loguru import logger

from app.llm import get_llm, LLMError
from app.meds.normalize import ALIASES, CLASSES, canonical_tokens

MEDS_VERSION = "drug-interactions/2026.06"

_DATA = Path(__file__).parent / "data" / "known_interactions.jsonl"
_SEVERITY_RANK = {"contraindicated": 4, "major": 3, "moderate": 2, "minor": 1, "none": 0}

_DISCLAIMER = (
    "Advisory only. This checks a curated reference set of common interactions, not "
    "a complete drug database — absence of a flag does NOT mean the combination is "
    "safe. A clinician or pharmacist must confirm against a full interaction "
    "reference before any prescribing decision."
)


@lru_cache(maxsize=1)
def _load_rules() -> list[dict]:
    rules: list[dict] = []
    with open(_DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rules.append(json.loads(line))
    return rules


@lru_cache(maxsize=1)
def _class_tokens() -> set[str]:
    """Every token that denotes a class (not a specific drug)."""
    out: set[str] = set()
    for cs in CLASSES.values():
        out |= cs
    return out


@lru_cache(maxsize=1)
def _known_generics() -> frozenset[str]:
    """Reference vocabulary of recognized generic drugs (for the unrecognized report)."""
    classes = _class_tokens()
    specific = {t for r in _load_rules() for t in (r["a"], r["b"]) if t not in classes}
    return frozenset(set(ALIASES.values()) | set(CLASSES.keys()) | specific)


def _match(tokens_a: set[str], tokens_b: set[str]) -> list[dict]:
    hits = []
    for r in _load_rules():
        a, b = r["a"], r["b"]
        if (a in tokens_a and b in tokens_b) or (a in tokens_b and b in tokens_a):
            hits.append(r)
    return hits


def check_interactions(medications: list[str], use_llm: bool = False) -> dict:
    """
    Check a medication list for pairwise interactions. Deterministic by default;
    pass use_llm=True to add an advisory LLM pass for *additional* candidates.
    """
    # Normalize + dedupe by generic, preserving first-seen display name.
    norm: list[tuple[str, set[str]]] = []
    generics: list[str] = []
    unrecognized: list[str] = []
    seen: set[str] = set()
    for raw in medications:
        generic, tokens = canonical_tokens(raw)
        if not generic or generic in seen:
            continue
        seen.add(generic)
        norm.append((generic, tokens))
        generics.append(generic)
        if generic not in _known_generics() and not (tokens - {generic}):
            unrecognized.append(generic)

    # One finding per drug pair — if several rules match (e.g. spironolactone is
    # both a named drug and a potassium-sparing diuretic), keep the most severe.
    by_pair: dict[frozenset[str], dict] = {}
    for (gen_a, tok_a), (gen_b, tok_b) in combinations(norm, 2):
        for rule in _match(tok_a, tok_b):
            key = frozenset((gen_a, gen_b))
            cand = {
                "drug_a": gen_a, "drug_b": gen_b,
                "severity": rule["severity"], "effect": rule["effect"],
                "management": rule["management"], "source": "curated-reference",
            }
            cur = by_pair.get(key)
            if cur is None or _SEVERITY_RANK[rule["severity"]] > _SEVERITY_RANK[cur["severity"]]:
                by_pair[key] = cand

    interactions = sorted(by_pair.values(), key=lambda i: -_SEVERITY_RANK[i["severity"]])
    highest = interactions[0]["severity"] if interactions else "none"

    llm_advisory: list[dict] = []
    llm_used = False
    if use_llm and len(generics) >= 2:
        llm_advisory, llm_used = _llm_additional(generics, interactions)

    return {
        "medications_checked": generics,
        "unrecognized": unrecognized,        # could not be matched — NOT cleared
        "interactions": interactions,
        "interaction_count": len(interactions),
        "highest_severity": highest,
        "has_contraindication": highest == "contraindicated",
        "llm_advisory": llm_advisory,        # additive candidates for a human to verify
        "advisory": _DISCLAIMER,
        "needs_human_review": True,          # always clinician/pharmacist confirmed
        "llm_used": llm_used,
        "model_version": MEDS_VERSION,
    }


_LLM_SYS = (
    "You are a clinical pharmacology assistant supporting a pharmacist. You are given "
    "a list of medications and the interactions ALREADY found by a reference checker. "
    "List only ADDITIONAL potentially-significant drug-drug interactions among the "
    "listed medications that are NOT already covered. Do NOT restate covered ones. Do "
    "NOT reassure or state anything is safe. If you are unsure or none apply, return an "
    "empty list. Respond ONLY with JSON: "
    '{"additional": [{"drug_a": "...", "drug_b": "...", "severity": '
    '"contraindicated|major|moderate|minor", "effect": "<short>"}]}.'
)


def _llm_additional(generics: list[str], known: list[dict]) -> tuple[list[dict], bool]:
    covered = "; ".join(f"{i['drug_a']}+{i['drug_b']}" for i in known) or "none"
    user = (f"MEDICATIONS: {', '.join(generics)}\n"
            f"ALREADY COVERED: {covered}\n"
            "List additional interactions not already covered.")
    try:
        from pydantic import BaseModel
        from typing import Literal

        class _Add(BaseModel):
            drug_a: str
            drug_b: str
            severity: Literal["contraindicated", "major", "moderate", "minor"]
            effect: str

        class _Resp(BaseModel):
            additional: list[_Add] = []

        resp: _Resp = get_llm().generate_json(
            messages=[{"role": "system", "content": _LLM_SYS},
                      {"role": "user", "content": user}],
            validate=_Resp, max_tokens=400,
        )
        out = [{**a.model_dump(), "source": "llm-advisory-unverified"}
               for a in resp.additional]
        return out, True
    except LLMError as exc:
        logger.warning(f"[meds] LLM advisory unavailable (deterministic result stands): {exc}")
        return [], False
    except Exception as exc:   # never let the advisory layer break the safety result
        logger.warning(f"[meds] LLM advisory error ignored: {exc}")
        return [], False
