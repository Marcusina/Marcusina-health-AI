"""
LLMClient — a thin, dependency-light wrapper over a self-hosted, OpenAI-compatible
chat-completions server (vLLM / Ollama / llama.cpp).

Design goals:
  * No external API, no paid key — base URL points at our own model server.
  * Portable across vLLM, Ollama, and llama.cpp (all expose /v1/chat/completions).
  * Resilient: transient connection errors / 5xx / timeouts are retried with
    backoff, so an on-demand cloud GPU that is still spinning up surfaces as a
    retryable LLMUnavailable rather than a hard failure.
  * JSON that actually parses: generate_json() requests JSON mode, robustly
    extracts the object even if the model wraps it in prose/fences, and runs one
    self-repair round before giving up. Optional Pydantic validation.

Synchronous on purpose — it is called from Celery worker processes.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Sequence, Type, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.llm.errors import LLMError, LLMUnavailable, LLMInvalidJSON

settings = get_settings()

Message = dict[str, str]                      # {"role": "...", "content": "..."}
T = TypeVar("T", bound=BaseModel)

_RETRYABLE_STATUS = {500, 502, 503, 504, 429}


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout if timeout is not None else settings.LLM_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {api_key or settings.LLM_API_KEY}"},
        )

    # ── public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Run a chat completion and return the assistant's text."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": list(messages),
            "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        }
        if stop:
            payload["stop"] = stop
        if extra:
            payload.update(extra)
        data = self._post("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed completion response: {exc}; body={data!r:.300}") from exc

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience: single user prompt (+ optional system) → text."""
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def generate_json(
        self,
        messages: Sequence[Message],
        *,
        schema: dict[str, Any] | None = None,
        validate: Type[T] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        repair: bool = True,
    ) -> dict[str, Any] | T:
        """
        Generate JSON. Requests JSON mode, extracts the object even if the model
        adds prose/fences, optionally validates against a Pydantic model, and
        runs one self-repair round on failure.

        Returns a `dict`, or a validated `validate` instance if `validate` is given.
        Raises LLMInvalidJSON if usable JSON can't be obtained.
        """
        msgs = list(messages)
        extra: dict[str, Any] = {"response_format": {"type": "json_object"}}
        # vLLM honours guided_json for grammar-level constraint; servers that
        # don't recognise it ignore the key. Best-effort, never required.
        if schema is not None:
            extra["guided_json"] = schema

        raw = self.chat(msgs, model=model, max_tokens=max_tokens, extra=extra)
        parsed, err = _try_parse_json(raw)

        if parsed is None and repair:
            logger.warning(f"[llm] JSON parse failed ({err}); attempting one repair round.")
            msgs = msgs + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                    f"That was not valid JSON ({err}). Reply with ONLY the corrected, "
                    f"complete JSON value. No prose, no markdown fences."},
            ]
            raw = self.chat(msgs, model=model, max_tokens=max_tokens, extra=extra)
            parsed, err = _try_parse_json(raw)

        if parsed is None:
            raise LLMInvalidJSON(f"Model did not return valid JSON: {err}", raw=raw)

        if validate is not None:
            try:
                return validate.model_validate(parsed)
            except ValidationError as ve:
                if not repair:
                    raise LLMInvalidJSON(f"JSON failed schema validation: {ve}", raw=raw) from ve
                logger.warning("[llm] JSON failed Pydantic validation; one repair round.")
                msgs = msgs + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                        f"The JSON did not match the required schema:\n{ve}\n"
                        f"Reply with ONLY corrected JSON matching the schema."},
                ]
                raw = self.chat(msgs, model=model, max_tokens=max_tokens, extra=extra)
                parsed, err = _try_parse_json(raw)
                if parsed is None:
                    raise LLMInvalidJSON(f"Repair produced invalid JSON: {err}", raw=raw) from ve
                try:
                    return validate.model_validate(parsed)
                except ValidationError as ve2:
                    raise LLMInvalidJSON(f"JSON still failed validation: {ve2}", raw=raw) from ve2

        return parsed

    def health(self) -> dict[str, Any]:
        """Liveness probe — lists models on the server. Cheap, no generation."""
        try:
            resp = self._client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            body = resp.json()
            ids = [m.get("id") for m in body.get("data", [])]
            return {"ok": True, "base_url": self.base_url, "models": ids,
                    "configured_model": self.model}
        except Exception as exc:  # noqa: BLE001 — health must never raise
            return {"ok": False, "base_url": self.base_url, "error": str(exc)}

    def close(self) -> None:
        self._client.close()

    # ── internals ───────────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(url, json=payload)
                if resp.status_code in _RETRYABLE_STATUS:
                    last_exc = LLMUnavailable(
                        f"{resp.status_code} from model server: {resp.text[:200]}")
                    self._backoff(attempt, f"HTTP {resp.status_code}")
                    continue
                if resp.status_code >= 400:
                    # Non-retryable client error — surface it directly.
                    raise LLMError(f"Model server {resp.status_code}: {resp.text[:300]}")
                return resp.json()
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    httpx.RemoteProtocolError, httpx.PoolTimeout) as exc:
                last_exc = exc
                self._backoff(attempt, type(exc).__name__)
        raise LLMUnavailable(
            f"Model server at {self.base_url} unreachable after "
            f"{self.max_retries + 1} attempts: {last_exc}")

    def _backoff(self, attempt: int, reason: str) -> None:
        if attempt < self.max_retries:
            delay = 2 ** attempt
            logger.warning(f"[llm] {reason}; retry {attempt + 1}/{self.max_retries} in {delay}s")
            time.sleep(delay)


# ── JSON extraction ──────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _try_parse_json(text: str) -> tuple[Any | None, str | None]:
    """
    Best-effort parse of model output into a JSON value. Handles bare JSON,
    ```json fenced blocks, and prose with an embedded object/array.
    Returns (value, None) on success or (None, error_message) on failure.
    """
    if text is None:
        return None, "empty response"
    candidate = text.strip()

    # 1. Direct parse.
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError:
        pass

    # 2. Fenced ```json ... ``` block.
    m = _FENCE_RE.search(candidate)
    if m:
        try:
            return json.loads(m.group(1)), None
        except json.JSONDecodeError:
            pass

    # 3. First balanced {...} or [...] span.
    span = _first_json_span(candidate)
    if span is not None:
        try:
            return json.loads(span), None
        except json.JSONDecodeError as exc:
            return None, f"embedded JSON did not parse: {exc}"

    return None, "no JSON object/array found in response"


def _first_json_span(text: str) -> str | None:
    """Return the first balanced {...} or [...] substring, respecting strings."""
    start = None
    opener = closer = ""
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if start is None:
            if ch in "{[":
                start, opener = i, ch
                closer = "}" if ch == "{" else "]"
                depth = 1
            continue
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ── process-wide singleton ───────────────────────────────────────────────────────

_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    """One client per process (reuses the HTTP connection pool)."""
    global _llm
    if _llm is None:
        _llm = LLMClient()
        logger.info(f"[llm] client ready → {_llm.base_url} (model={_llm.model})")
    return _llm
