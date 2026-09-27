"""Open models via any OpenAI-compatible chat endpoint (Ollama, vLLM, SGLang, ...)."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from ercot_bench.models.base import Completion, ModelClient, RateLimited


class OpenAICompatClient(ModelClient):
    backend = "openai-compat"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 4096, timeout_s: float = 600,
                 use_n: bool = False, max_retries: int = 4, extra_body: dict | None = None, **_: Any):
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY") or "not-needed"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        # vLLM supports n>1 in one request; Ollama ignores n, so default to one request per sample.
        self.use_n = use_n
        self.max_retries = max_retries
        self.extra_body = extra_body or {}

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "max_tokens": self.max_tokens}

    async def _request(self, client: httpx.AsyncClient, system_prompt: str, user_prompt: str, n: int) -> list[Completion]:
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "n": n,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                **self.extra_body}
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                r = await client.post(f"{self.base_url}/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {self.api_key}"})
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                latency = time.time() - t0
                usage = data.get("usage") or {}
                choices = data.get("choices") or []
                per = len(choices) or 1
                out = []
                for c in choices:
                    msg = c.get("message") or {}
                    text = msg.get("content") or ""
                    # some servers split reasoning out; keep it so <think> handling is uniform
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                    if reasoning and reasoning.strip():
                        text = f"<think>{reasoning}</think>\n{text}"
                    out.append(Completion(
                        text=text, input_tokens=usage.get("prompt_tokens"),
                        output_tokens=(usage.get("completion_tokens") or 0) // per or None, latency_s=latency,
                        meta={"finish_reason": c.get("finish_reason"), "temperature": self.temperature}))
                return out
            except (httpx.HTTPError, ValueError) as e:
                if attempt == self.max_retries:
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                        raise RateLimited(str(e)) from e
                    return [Completion(text="", latency_s=time.time() - t0, error=f"{type(e).__name__}: {e}")] * n
                await asyncio.sleep(2 * (2 ** attempt))
        raise AssertionError("unreachable")

    async def generate(self, system_prompt: str, user_prompt: str, n: int = 1) -> list[Completion]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            if self.use_n:
                return await self._request(client, system_prompt, user_prompt, n)
            out: list[Completion] = []
            for _ in range(n):
                out += await self._request(client, system_prompt, user_prompt, 1)
            return out
