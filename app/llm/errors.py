"""LLM lane exceptions — distinct types so callers can react appropriately."""

from __future__ import annotations


class LLMError(Exception):
    """Base for any failure in the local LLM lane."""


class LLMUnavailable(LLMError):
    """
    The model server could not be reached or did not respond in time.

    For the on-demand cloud GPU this is the expected error when the GPU is spun
    down — callers (Celery tasks) should retry with backoff, not fail the job.
    """


class LLMInvalidJSON(LLMError):
    """The model did not return JSON that parses / validates, even after repair."""

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw
