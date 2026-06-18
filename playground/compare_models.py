"""
Side-by-side model comparison for the AI capabilities.

Runs the same inputs (triage, misinfo RAG, SOAP) through each model so you can
judge output quality before choosing one. Switches the LLM by setting the
singleton's model — no .env change needed.

    .venv/Scripts/python -m playground.compare_models llama3:latest granite4.1:8b
"""

from __future__ import annotations

import sys
import time

from app.llm import get_llm
from app.safety import assess_triage
from app.rag import check_claim
from app.clinical import generate_soap

TRANSCRIPT = (
    "Doctor: What brings you in today?\n"
    "Patient: I've had a cough for about three days and a mild fever. No chest pain.\n"
    "Doctor: Any shortness of breath? Patient: No. I took some paracetamol.\n"
    "Doctor: Temperature is 37.9, chest is clear. Looks like a viral infection. "
    "Rest, fluids, and paracetamol as needed. Come back if it worsens."
)

CASES = [
    ("TRIAGE  (no red flag → LLM urgency)",
     lambda: assess_triage("persistent moderate headache for three days, mild nausea", use_llm=True),
     lambda r: f"{r['urgency_level']}  | {r['reasoning'][:140]}"),
    ("MISINFO (RAG: retrieve + judge)  claim='drinking bleach cures COVID-19'",
     lambda: check_claim("drinking bleach cures COVID-19", k=3),
     lambda r: f"{r['verdict']} (conf {r['confidence']}) | cites {len(r['citations'])} | {r['rationale'][:140]}"),
    ("MISINFO  claim='regular handwashing reduces infection'",
     lambda: check_claim("regular handwashing reduces the spread of infection", k=3),
     lambda r: f"{r['verdict']} (conf {r['confidence']}) | {r['rationale'][:120]}"),
    ("SOAP    (note + entities + ICD)",
     lambda: generate_soap(TRANSCRIPT),
     lambda r: (f"degraded={r['degraded']} | ICD={r['icd_suggestions']}\n"
                f"      Assessment: {r['soap_note']['assessment'][:160]}\n"
                f"      Meds={r['extracted_entities']['medications']} "
                f"Symptoms={r['extracted_entities']['symptoms']}")),
]


def run(models: list[str]) -> None:
    llm = get_llm()
    for name, fn, fmt in CASES:
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        for model in models:
            llm.model = model
            t = time.perf_counter()
            try:
                out = fmt(fn())
            except Exception as exc:  # noqa: BLE001
                out = f"ERROR: {exc}"
            print(f"\n  [{model}]  ({time.perf_counter() - t:.1f}s)\n      {out}")


if __name__ == "__main__":
    models = sys.argv[1:] or ["llama3:latest", "granite4.1:8b"]
    print(f"Comparing: {models}  (first call per model includes load time)")
    run(models)
