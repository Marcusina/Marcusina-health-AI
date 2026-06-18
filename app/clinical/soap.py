"""
SOAP note generation from a consultation transcript.

The LLM produces the four SOAP sections AND extracts clinical entities in one
JSON-constrained call. Diagnoses/symptoms are then mapped to ICD-10 codes
deterministically (config/icd_map.json) — the model never invents codes.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from app.llm import get_llm, LLMError
from app.utils.config_loader import get_icd_map


def _coerce_str_list(v) -> list[str]:
    """
    Entity lists come back in varied shapes across models — a bare string
    ('Temperature: 37.9C'), a dict ({'Temperature': '37.9C'}), or a list mixing
    strings and dicts. Normalize all of them to list[str] so a well-formed note
    isn't rejected over formatting.
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, dict):
        return [f"{k}: {val}" for k, val in v.items()]
    if isinstance(v, list):
        out: list[str] = []
        for item in v:
            if item is None:
                continue
            if isinstance(item, dict):
                out.extend(f"{k}: {val}" for k, val in item.items())
            else:
                out.append(str(item))
        return out
    return [str(v)]

SOAP_VERSION = "soap-llm/2026.06"

_SYS = (
    "You are a clinical documentation assistant. From the consultation transcript, "
    "produce a SOAP note and extract clinical entities. Use ONLY information present "
    "in the transcript — do not invent findings, medications, or diagnoses. If a SOAP "
    "section has no information, write a brief 'Not documented.' Keep each section "
    "concise and clinical. This note will be reviewed and signed by a clinician.\n"
    "Respond ONLY with JSON of this shape:\n"
    '{"subjective": "...", "objective": "...", "assessment": "...", "plan": "...", '
    '"medications": [..], "diagnoses": [..], "symptoms": [..], '
    '"procedures": [..], "vitals": [..]}'
)


class SoapNote(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    medications: list[str] = Field(default_factory=list)
    diagnoses: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    vitals: list[str] = Field(default_factory=list)

    @field_validator("medications", "diagnoses", "symptoms", "procedures", "vitals",
                     mode="before")
    @classmethod
    def _coerce(cls, v):
        return _coerce_str_list(v)


# Negation cues — if one appears before a matched keyword, the finding is a
# pertinent NEGATIVE ("no chest pain") and must not produce an ICD code.
_NEGATION_CUES = (
    "no ", "not ", "without ", "denies", "denied", "negative for", "negative",
    "absent", "absence of", "no evidence of", "ruled out", "r/o ", "non-", "free of",
)


def _negated_before(prefix: str) -> bool:
    return any(cue in prefix for cue in _NEGATION_CUES)


def suggest_icd(diagnoses: list[str], symptoms: list[str], limit: int = 5) -> list[str]:
    """
    Map extracted terms to ICD-10 codes via the config map (deterministic).
    Skips negated findings — "no chest pain" must not yield the chest-pain code.
    """
    icd_map = get_icd_map()
    codes: list[str] = []
    for term in [*diagnoses, *symptoms]:
        t = term.lower()
        for keyword, code in icd_map.items():
            idx = t.find(keyword)
            if idx == -1 or code in codes:
                continue
            if _negated_before(t[:idx]):
                continue                    # pertinent negative — skip
            codes.append(code)
    return codes[:limit]


def generate_soap(transcript: str, *, patient_id: str | None = None,
                  specialty: str | None = None) -> dict:
    """
    Generate a SOAP note. Returns a dict with the note, extracted entities, ICD
    suggestions, and provenance flags. Falls back to a degraded rule-based draft
    if the LLM is unavailable (never raises for an LLM outage).
    """
    user = f"Transcript:\n{transcript.strip()}"
    if specialty:
        user += f"\n\n(Consultation specialty: {specialty}.)"

    try:
        note: SoapNote = get_llm().generate_json(
            messages=[{"role": "system", "content": _SYS},
                      {"role": "user", "content": user}],
            validate=SoapNote, max_tokens=1200,
        )
        entities = {
            "medications": note.medications, "diagnoses": note.diagnoses,
            "symptoms": note.symptoms, "procedures": note.procedures, "vitals": note.vitals,
        }
        return {
            "soap_note": {"subjective": note.subjective, "objective": note.objective,
                          "assessment": note.assessment, "plan": note.plan},
            "extracted_entities": entities,
            "icd_suggestions": suggest_icd(note.diagnoses, note.symptoms),
            "llm_used": True, "degraded": False, "model_version": SOAP_VERSION,
        }
    except LLMError as exc:
        logger.warning(f"[clinical] SOAP LLM unavailable, returning degraded draft: {exc}")
        return _degraded_soap(transcript)


def _degraded_soap(transcript: str) -> dict:
    """Minimal, clearly-marked fallback when the LLM is down — clinician must complete it."""
    excerpt = transcript.strip()[:400]
    return {
        "soap_note": {
            "subjective": f"[AUTO-DRAFT — LLM unavailable] Transcript excerpt: {excerpt}…",
            "objective": "Not documented (automated extraction unavailable).",
            "assessment": "Not documented (automated extraction unavailable).",
            "plan": "Not documented (automated extraction unavailable).",
        },
        "extracted_entities": {"medications": [], "diagnoses": [], "symptoms": [],
                               "procedures": [], "vitals": []},
        "icd_suggestions": [],
        "llm_used": False, "degraded": True, "model_version": SOAP_VERSION,
    }
