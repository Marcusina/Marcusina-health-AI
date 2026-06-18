"""
Local LLM lane.

Wraps a self-hosted, OpenAI-compatible model server (vLLM on a cloud GPU in
production; Ollama / llama.cpp on CPU for dev). There is no external API and no
paid key — the AI service talks to a model server inside our own deployment, so
no patient data ever leaves our infrastructure.

Moving from a local CPU model to a cloud GPU, or swapping the model entirely, is
a change of LLM_BASE_URL / LLM_MODEL in config — never a code change, because
every backend (vLLM, Ollama, llama.cpp) speaks the same /v1/chat/completions
interface.

Public surface:
    get_llm()            -> process-wide singleton LLMClient
    LLMClient.chat(...)  -> free-text generation
    LLMClient.generate_json(...) -> JSON-constrained generation (+ optional Pydantic validation)
    LLMError / LLMUnavailable / LLMInvalidJSON
"""

from app.llm.client import LLMClient, get_llm
from app.llm.errors import LLMError, LLMUnavailable, LLMInvalidJSON

__all__ = [
    "LLMClient",
    "get_llm",
    "LLMError",
    "LLMUnavailable",
    "LLMInvalidJSON",
]
