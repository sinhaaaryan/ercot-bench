"""Optional: Anthropic Messages API backend (requires ANTHROPIC_API_KEY and `uv sync --extra api`)."""

from __future__ import annotations

import time
from typing import Any

from ercot_bench.models.base import Completion, ModelClient, RateLimited


class AnthropicAPIClient(ModelClient):
    backend = "anthropic-api"

    def __init__(self, model: str = "claude-sonnet-5", temperature: float = 1.0, max_tokens: int = 4096, **_: Any):
        import anthropic

        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model, "temperature": self.temperature}

    async def generate(self, system_prompt: str, user_prompt: str, n: int = 1) -> list[Completion]:
        import anthropic

        out = []
        for _ in range(n):
            t0 = time.time()
            try:
                msg = await self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens, temperature=self.temperature, system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}])
            except anthropic.RateLimitError as e:
                raise RateLimited(str(e)) from e
            text = "".join(b.text for b in msg.content if b.type == "text")
            out.append(Completion(text=text, input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                                  latency_s=time.time() - t0, meta={"stop_reason": msg.stop_reason}))
        return out
