"""
Unit tests for the local LLM lane (app/llm).

These run with NO model server: the pure JSON-extraction helpers are tested
directly, and generate_json() is tested by stubbing LLMClient.chat so the
parse / repair / validate logic is exercised deterministically.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.llm.client import LLMClient, _try_parse_json, _first_json_span
from app.llm.errors import LLMInvalidJSON


# ── pure JSON extraction ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('  {"a": 1}\n', {"a": 1}),
    ('```json\n{"a": 1, "b": [2,3]}\n```', {"a": 1, "b": [2, 3]}),
    ('Sure! Here is the result:\n{"a": 1}\nHope that helps.', {"a": 1}),
    ('[1, 2, 3]', [1, 2, 3]),
    ('{"text": "a } brace in a string", "n": 2}', {"text": "a } brace in a string", "n": 2}),
])
def test_try_parse_json_ok(text, expected):
    value, err = _try_parse_json(text)
    assert err is None
    assert value == expected


@pytest.mark.parametrize("text", ["not json at all", "", "{broken", "<html>nope</html>"])
def test_try_parse_json_fail(text):
    value, err = _try_parse_json(text)
    assert value is None
    assert err


def test_first_json_span_respects_strings_and_nesting():
    s = 'prefix {"outer": {"inner": "}{ tricky"}} suffix'
    span = _first_json_span(s)
    assert span == '{"outer": {"inner": "}{ tricky"}}'


# ── generate_json with a stubbed chat ─────────────────────────────────────────

class _Probe(BaseModel):
    urgency: str
    reason: str


def _client_returning(*responses: str) -> LLMClient:
    """Build a client whose .chat yields the given responses in order."""
    client = LLMClient(base_url="http://127.0.0.1:9/v1", model="test")
    queue = list(responses)

    def fake_chat(*args, **kwargs):
        assert queue, "chat called more times than canned responses"
        return queue.pop(0)

    client.chat = fake_chat  # type: ignore[assignment]
    return client


def test_generate_json_direct():
    client = _client_returning('{"urgency": "urgent", "reason": "chest pain"}')
    out = client.generate_json([{"role": "user", "content": "x"}])
    assert out == {"urgency": "urgent", "reason": "chest pain"}


def test_generate_json_from_prose():
    client = _client_returning('Here you go:\n```json\n{"urgency": "routine", "reason": "mild"}\n```')
    out = client.generate_json([{"role": "user", "content": "x"}])
    assert out["urgency"] == "routine"


def test_generate_json_repairs_then_succeeds():
    client = _client_returning(
        "I cannot output JSON sorry",                       # bad first attempt
        '{"urgency": "emergency", "reason": "crushing chest pain"}',  # repair
    )
    out = client.generate_json([{"role": "user", "content": "x"}])
    assert out["urgency"] == "emergency"


def test_generate_json_validates_with_pydantic():
    client = _client_returning('{"urgency": "routine", "reason": "ok"}')
    out = client.generate_json([{"role": "user", "content": "x"}], validate=_Probe)
    assert isinstance(out, _Probe)
    assert out.urgency == "routine"


def test_generate_json_repairs_validation_error():
    client = _client_returning(
        '{"urgency": "routine"}',                            # missing required field
        '{"urgency": "routine", "reason": "added on repair"}',
    )
    out = client.generate_json([{"role": "user", "content": "x"}], validate=_Probe)
    assert out.reason == "added on repair"


def test_generate_json_unrepairable_raises():
    client = _client_returning("garbage", "still garbage")
    with pytest.raises(LLMInvalidJSON):
        client.generate_json([{"role": "user", "content": "x"}])


def test_health_down_server_returns_not_ok():
    # Port 9 (discard) — connection refused fast; health must never raise.
    client = LLMClient(base_url="http://127.0.0.1:9/v1", model="test", timeout=2.0)
    h = client.health()
    assert h["ok"] is False
    assert "error" in h
