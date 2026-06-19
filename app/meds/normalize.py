"""
Drug-name normalization for the interaction checker.

Two small reference maps, deliberately seed-sized and easy to audit:

  * ALIASES  — common brand names / spellings → canonical generic ingredient.
  * CLASSES  — generic ingredient → the drug classes it belongs to, so a
               class-level rule ("NSAID + warfarin") matches every NSAID without
               enumerating each one.

`canonical_tokens()` turns a free-text medication string into the set of tokens
an interaction rule can match against: its generic name plus every class it's in.
A rule fires when one of its two tokens is in drug A's token set and the other is
in drug B's. This is intentionally simple and deterministic — the authoritative
safety layer must be inspectable, not a black box.

This is a curated seed, NOT a complete drug database. Unknown drugs pass through
normalized-but-unclassified; the checker reports them as such rather than
implying they were cleared.
"""

from __future__ import annotations

import re

# brand / common alias  ->  canonical generic
ALIASES: dict[str, str] = {
    "coumadin": "warfarin", "jantoven": "warfarin",
    "advil": "ibuprofen", "motrin": "ibuprofen", "nurofen": "ibuprofen",
    "aleve": "naproxen", "naprosyn": "naproxen",
    "aspirin": "aspirin", "asa": "aspirin", "acetylsalicylic acid": "aspirin",
    "tylenol": "acetaminophen", "paracetamol": "acetaminophen",
    "zoloft": "sertraline", "prozac": "fluoxetine", "lexapro": "escitalopram",
    "paxil": "paroxetine", "celexa": "citalopram",
    "ultram": "tramadol", "viagra": "sildenafil", "revatio": "sildenafil",
    "nitroglycerin": "nitroglycerin", "gtn": "nitroglycerin",
    "isosorbide": "isosorbide", "imdur": "isosorbide",
    "glucophage": "metformin", "lipitor": "atorvastatin", "zocor": "simvastatin",
    "crestor": "rosuvastatin", "norvasc": "amlodipine",
    "prinivil": "lisinopril", "zestril": "lisinopril",
    "cozaar": "losartan", "aldactone": "spironolactone",
    "lanoxin": "digoxin", "cordarone": "amiodarone", "pacerone": "amiodarone",
    "calan": "verapamil", "isoptin": "verapamil",
    "plavix": "clopidogrel", "prilosec": "omeprazole", "losec": "omeprazole",
    "eskalith": "lithium", "lithobid": "lithium",
    "rheumatrex": "methotrexate", "trexall": "methotrexate",
    "bactrim": "trimethoprim", "septra": "trimethoprim", "co-trimoxazole": "trimethoprim",
    "cipro": "ciprofloxacin", "biaxin": "clarithromycin",
    "diflucan": "fluconazole", "nizoral": "ketoconazole",
    "theo-dur": "theophylline", "valium": "diazepam", "xanax": "alprazolam",
    "ativan": "lorazepam", "oxycontin": "oxycodone", "vicodin": "hydrocodone",
    "hctz": "hydrochlorothiazide", "microzide": "hydrochlorothiazide",
    "nardil": "phenelzine", "parnate": "tranylcypromine", "marplan": "isocarboxazid",
    "potassium chloride": "potassium", "kcl": "potassium", "k-dur": "potassium",
    "klor-con": "potassium", "slow-k": "potassium",
}

# generic ingredient -> classes it belongs to (drives class-level rules)
CLASSES: dict[str, set[str]] = {
    "ibuprofen": {"nsaid"}, "naproxen": {"nsaid"}, "diclofenac": {"nsaid"},
    "aspirin": {"nsaid", "antiplatelet"},
    "sertraline": {"ssri"}, "fluoxetine": {"ssri"}, "escitalopram": {"ssri"},
    "paroxetine": {"ssri"}, "citalopram": {"ssri"},
    "atorvastatin": {"statin"}, "simvastatin": {"statin"}, "rosuvastatin": {"statin"},
    "lisinopril": {"ace_inhibitor"}, "enalapril": {"ace_inhibitor"}, "ramipril": {"ace_inhibitor"},
    "losartan": {"arb"},
    "spironolactone": {"potassium_sparing_diuretic"},
    "nitroglycerin": {"nitrate"}, "isosorbide": {"nitrate"},
    "clarithromycin": {"macrolide", "cyp3a4_inhibitor"},
    "erythromycin": {"macrolide", "cyp3a4_inhibitor"},
    "ketoconazole": {"azole_antifungal", "cyp3a4_inhibitor"},
    "fluconazole": {"azole_antifungal", "cyp3a4_inhibitor"},
    "diazepam": {"benzodiazepine"}, "alprazolam": {"benzodiazepine"},
    "lorazepam": {"benzodiazepine"},
    "oxycodone": {"opioid"}, "hydrocodone": {"opioid"}, "tramadol": {"opioid"},
    "morphine": {"opioid"}, "fentanyl": {"opioid"},
    "hydrochlorothiazide": {"thiazide"},
    "phenelzine": {"maoi"}, "tranylcypromine": {"maoi"}, "isocarboxazid": {"maoi"},
    "selegiline": {"maoi"},
}

_DOSE = re.compile(
    r"\b(\d+(\.\d+)?\s*(mg|mcg|g|ml|units?|iu|tab(let)?s?|cap(sule)?s?|"
    r"od|bd|tds|qid|prn|daily|once|twice))\b", re.I)


# every string we can resolve to a generic: alias keys, alias targets, class keys
_KNOWN: set[str] = set(ALIASES) | set(ALIASES.values()) | set(CLASSES)


def _resolve(token: str) -> str | None:
    if token in ALIASES:
        return ALIASES[token]
    if token in _KNOWN:
        return token
    return None


def normalize_name(raw: str) -> str:
    """
    Lowercase, strip dosage/frequency noise and punctuation, map brand→generic.
    Handles "<generic> <salt/form>" (e.g. "isosorbide mononitrate", "metformin
    hydrochloride") by scanning for a recognized word when the whole string isn't
    a direct match.
    """
    s = raw.lower().strip()
    s = _DOSE.sub(" ", s)
    s = re.sub(r"[^a-z\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    direct = _resolve(s)
    if direct:
        return direct
    # fall back to the first recognized word (drops salt/form suffixes etc.)
    for word in s.split():
        hit = _resolve(word)
        if hit:
            return hit
    return s


def canonical_tokens(raw: str) -> tuple[str, set[str]]:
    """Return (generic_name, {generic} ∪ classes) — the tokens a rule matches on."""
    generic = normalize_name(raw)
    return generic, {generic} | CLASSES.get(generic, set())
