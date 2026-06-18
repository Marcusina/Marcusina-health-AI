"""
Grounded misinformation check: retrieve trusted evidence, then have the LLM judge
the claim *against that evidence only*.

Verdicts:
  supported        — evidence supports the claim (reliable)
  contradicted     — evidence contradicts the claim (likely misinfo)
  unsupported      — evidence does not address the claim (can't verify)
  not_health_claim — not a checkable health claim
  unverified       — LLM unavailable; needs human review (fail-safe)

Output is ADVISORY: anything flagged goes to a human reviewer, never an automatic
removal (see docs/AI-PLATFORM-DESIGN.md §6).
"""

from __future__ import annotations

from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from app.llm import get_llm, LLMError
from app.rag.retriever import get_retriever, Retriever

MISINFO_VERSION = "misinfo-rag/2026.06"

_SYS = (
    "You are a health-information fact-checker. You are given a CLAIM and a numbered "
    "list of EVIDENCE passages from trusted sources (WHO, CDC, etc.). Judge the claim "
    "ONLY against the provided evidence — do not use outside knowledge and do not guess.\n"
    "- If the evidence supports the claim, verdict='supported'.\n"
    "- If the evidence contradicts the claim, verdict='contradicted'.\n"
    "- If the evidence does not address the claim, verdict='unsupported'.\n"
    "- If the input is not a checkable health claim, verdict='not_health_claim'.\n"
    "Cite the evidence numbers you used in citation_ids. Respond ONLY with JSON: "
    '{"verdict": "supported|contradicted|unsupported|not_health_claim", '
    '"confidence": 0.0-1.0, "rationale": "<one sentence>", "citation_ids": [<int>, ...]}.'
)


class _Verdict(BaseModel):
    verdict: Literal["supported", "contradicted", "unsupported", "not_health_claim"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    citation_ids: list[int] = Field(default_factory=list)


def _judge(claim: str, evidence: list[dict]) -> Optional[_Verdict]:
    block = "\n".join(
        f"[{i + 1}] ({d['source']}) {d['text']}" for i, d in enumerate(evidence)
    )
    try:
        return get_llm().generate_json(
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": f"CLAIM:\n{claim}\n\nEVIDENCE:\n{block}"},
            ],
            validate=_Verdict,
            max_tokens=300,
        )
    except LLMError as exc:
        logger.warning(f"[rag] misinfo judge unavailable, failing to human review: {exc}")
        return None


def _citation(d: dict) -> dict:
    snippet = d["text"]
    return {
        "source": d.get("source", "unknown"),
        "url": d.get("url"),
        "snippet": snippet[:240] + ("…" if len(snippet) > 240 else ""),
        "score": d.get("score"),
    }


def check_claim(text: str, k: int = 4, retriever: Retriever | None = None) -> dict:
    """Retrieve evidence, judge the claim against it, return an advisory verdict."""
    r = retriever or get_retriever()
    evidence = r.search(text, k=k)

    # No evidence retrieved — corpus gap; can't verify.
    if not evidence:
        return _result(text, "unsupported", 0.0,
                       "No relevant trusted-source evidence found for this claim.",
                       citations=[], llm_used=False)

    j = _judge(text, evidence)

    # LLM unavailable — fail safe to human review, hand the reviewer the evidence.
    if j is None:
        return _result(text, "unverified", 0.0,
                       "Automated check unavailable; routed for human review.",
                       citations=[_citation(d) for d in evidence], llm_used=False)

    # Map the cited evidence numbers (1-based) back to citation objects.
    cited = [evidence[i - 1] for i in j.citation_ids if 1 <= i <= len(evidence)]
    if not cited and j.verdict in ("supported", "contradicted"):
        cited = evidence[:1]   # ensure a flagged/affirmed verdict shows its basis
    return _result(text, j.verdict, j.confidence, j.rationale,
                   citations=[_citation(d) for d in cited], llm_used=True)


def _result(claim: str, verdict: str, confidence: float, rationale: str,
            *, citations: list[dict], llm_used: bool) -> dict:
    needs_review = verdict in ("contradicted", "unsupported", "unverified")
    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": round(float(confidence), 3),
        "rationale": rationale,
        "flag": needs_review,                 # convenience for the backend gate
        "needs_human_review": needs_review,   # advisory only — never auto-remove
        "citations": citations,
        "llm_used": llm_used,
        "model_version": MISINFO_VERSION,
    }
