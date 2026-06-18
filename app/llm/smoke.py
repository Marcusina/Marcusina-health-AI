"""
Smoke test for the local LLM lane. Run it once a model server is up:

    # dev, with Ollama:
    ollama serve &           # if not already running
    ollama pull mistral      # one-time
    python -m app.llm.smoke

It (1) probes the server, (2) does a free-text completion, and (3) does a
JSON-constrained generation validated against a Pydantic model — exercising the
exact paths the SOAP / triage / misinfo tasks will use.

Configure the target with LLM_BASE_URL / LLM_MODEL in .env. Nothing here calls an
external API or needs a paid key.
"""

from __future__ import annotations

import sys

from pydantic import BaseModel, Field

from app.llm import get_llm, LLMError


class _UrgencyProbe(BaseModel):
    urgency: str = Field(description="one of: emergency, urgent, routine")
    reason: str


def main() -> int:
    llm = get_llm()

    print(f"[1/3] health check → {llm.base_url}")
    h = llm.health()
    print("      ", h)
    if not h["ok"]:
        print("\nModel server not reachable. Start one (e.g. `ollama serve`) and set "
              "LLM_BASE_URL / LLM_MODEL, then re-run.")
        return 1

    try:
        print("\n[2/3] free-text completion")
        text = llm.complete(
            "In one sentence, what is a SOAP note in clinical documentation?",
            system="You are a concise medical documentation assistant.",
            max_tokens=120,
        )
        print("      →", text.strip())

        print("\n[3/3] JSON-constrained + Pydantic validation")
        result = llm.generate_json(
            messages=[
                {"role": "system", "content":
                    "You are a triage assistant. Respond ONLY with JSON matching: "
                    '{"urgency": "emergency|urgent|routine", "reason": "<short>"}.'},
                {"role": "user", "content": "Patient reports crushing chest pain radiating to the left arm."},
            ],
            validate=_UrgencyProbe,
            max_tokens=200,
        )
        print("      →", result.model_dump())
    except LLMError as exc:
        print(f"\nLLM call failed: {exc}")
        return 1

    print("\nOK — local LLM lane is working end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
