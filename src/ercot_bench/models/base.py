"""Shared model-client interface."""

from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel, Field


class Completion(BaseModel):
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_s: float
    error: str | None = None  # backend-level failure (not a model answer)
    meta: dict[str, Any] = Field(default_factory=dict)


class RateLimited(Exception):
    """Raised when the backend reports persistent usage/rate limits; the runner stops cleanly."""


class BackendUnavailable(Exception):
    """Raised on non-retryable backend failures (e.g. CLI not logged in); the runner stops cleanly."""


class ModelClient(abc.ABC):
    backend: str
    model: str

    @abc.abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str, n: int = 1) -> list[Completion]:
        ...

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model}


def make_client(backend: str, model: str, **kw) -> ModelClient:
    if backend == "claude-cli":
        from ercot_bench.models.claude_cli import ClaudeCLIClient
        return ClaudeCLIClient(model=model, **kw)
    if backend == "openai-compat":
        from ercot_bench.models.openai_compat import OpenAICompatClient
        return OpenAICompatClient(model=model, **kw)
    if backend == "anthropic-api":
        from ercot_bench.models.anthropic_api import AnthropicAPIClient
        return AnthropicAPIClient(model=model, **kw)
    raise ValueError(f"unknown backend {backend!r} (claude-cli | openai-compat | anthropic-api)")
