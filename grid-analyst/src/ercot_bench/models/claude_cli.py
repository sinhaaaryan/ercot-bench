"""Frontier models via Claude Code headless mode (`claude -p`), using the logged-in subscription.

Claude Code must behave as a plain single-turn model, not an agent. Flags verified against
`claude --help` for Claude Code 2.1.281:
  --system-prompt <p>        replaces Claude Code's default system prompt
  --tools ""                 disables all built-in tools
  --max-turns 1              single turn (accepted by the CLI though not listed in --help)
  --strict-mcp-config        no MCP servers (none given via --mcp-config)
  --setting-sources ""       ignore user/project/local settings (no hooks, plugins, permissions)
  --disable-slash-commands   no skills
  --no-session-persistence   don't write session files
  --output-format json       single JSON result; response text in .result
cwd is a fresh empty temp dir, so the repo and data are not visible (and there are no tools anyway).

Known, unavoidable residue (see README): the CLI still prepends a short Agent-SDK identity line,
an environment block (cwd/platform/date) and account context to the system prompt. `--bare` would
remove it but requires an API key, which this backend deliberately doesn't assume.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
from typing import Any

from ercot_bench.models.base import BackendUnavailable, Completion, ModelClient, RateLimited

log = logging.getLogger("ercot_bench.models.claude_cli")

RATE_LIMIT_MARKERS = ("usage limit", "rate limit", "rate_limit", "limit reached", "429", "overloaded", "529",
                      "too many requests")


def base_args(system_prompt: str, model: str, output_format: str = "json", effort: str | None = None,
              thinking: bool = True) -> list[str]:
    exe = shutil.which("claude") or "claude"
    args = [exe, "-p", "--system-prompt", system_prompt, "--tools", "", "--max-turns", "1",
            "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
            "--no-session-persistence", "--model", model, "--output-format", output_format]
    if output_format == "stream-json":
        args.append("--verbose")
    if effort:
        args += ["--effort", effort]
    if not thinking:  # verified: output_tokens_details.thinking_tokens == 0 with this setting
        args += ["--settings", '{"alwaysThinkingEnabled": false}']
    return args


async def _run(args: list[str], stdin_text: str, timeout_s: float) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory(prefix="ercot-bench-cc-") as cwd:
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=cwd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin_text.encode()), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "", f"timeout after {timeout_s}s"
        return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


def _is_rate_limited(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in RATE_LIMIT_MARKERS)


class ClaudeCLIClient(ModelClient):
    backend = "claude-cli"

    def __init__(self, model: str = "sonnet", timeout_s: float = 300, max_retries: int = 4,
                 backoff_s: float = 30, effort: str | None = None, thinking: bool = True, **_: Any):
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.effort = effort
        self.thinking = thinking

    def describe(self) -> dict[str, Any]:
        model = self.model if self.thinking else f"{self.model} (no thinking)"
        return {"backend": self.backend, "model": model, "temperature": "cli-default", "effort": self.effort,
                "thinking": self.thinking}

    async def _one(self, system_prompt: str, user_prompt: str) -> Completion:
        args = base_args(system_prompt, self.model, "json", self.effort, self.thinking)
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            rc, out, err = await _run(args, user_prompt, self.timeout_s)
            latency = time.time() - t0
            data: dict[str, Any] = {}
            try:
                data = json.loads(out) if out.strip() else {}
            except json.JSONDecodeError:
                pass
            text = data.get("result") or ""
            is_error = bool(data.get("is_error")) or rc != 0 or not data
            if not is_error:
                usage = data.get("usage") or {}
                in_tok = sum(usage.get(k) or 0 for k in
                             ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")) or None
                return Completion(
                    text=text, input_tokens=in_tok, output_tokens=usage.get("output_tokens"),
                    cost_usd=data.get("total_cost_usd"), latency_s=latency,
                    meta={"num_turns": data.get("num_turns"), "subtype": data.get("subtype"),
                          "session_id": data.get("session_id"), "temperature": "cli-default",
                          "model_usage": list((data.get("modelUsage") or {}).keys())})
            msg = (text or err or out)[:500]
            if "not logged in" in msg.lower() or "/login" in msg:
                raise BackendUnavailable(f"claude CLI is not logged in ({msg[:120]}). Run `claude` and /login, then rerun.")
            if rc == -1 and attempt >= 1:
                raise BackendUnavailable(f"claude CLI timed out twice ({msg[:120]}); check `claude -p hi` works "
                                         "(a CLI auto-update can require re-approving Keychain access on macOS).")
            if _is_rate_limited(msg) and attempt < self.max_retries:
                wait = self.backoff_s * (2 ** attempt)
                log.warning("claude CLI rate/usage limited (%s); backing off %.0fs", msg[:120], wait)
                await asyncio.sleep(wait)
                continue
            if _is_rate_limited(msg):
                raise RateLimited(msg)
            if attempt < min(self.max_retries, 2):
                await asyncio.sleep(3)
                continue
            return Completion(text="", latency_s=latency, error=f"rc={rc}: {msg}")
        raise RateLimited("exhausted retries")

    async def generate(self, system_prompt: str, user_prompt: str, n: int = 1) -> list[Completion]:
        return [await self._one(system_prompt, user_prompt) for _ in range(n)]


async def check_single_turn(system_prompt: str, user_prompt: str, model: str, timeout_s: float = 300) -> dict:
    """Run once with stream-json and verify: no tools available, no tool calls, exactly one assistant turn."""
    rc, out, err = await _run(base_args(system_prompt, model, "stream-json"), user_prompt, timeout_s)
    events = []
    for line in out.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    assistant = [e for e in events if e.get("type") == "assistant"]
    msg_ids = {e["message"].get("id") for e in assistant}
    tool_uses = [c for e in assistant for c in e["message"].get("content", []) if c.get("type") == "tool_use"]
    tool_results = [e for e in events if e.get("type") == "user"]
    result = next((e for e in events if e.get("type") == "result"), {})
    text = "".join(c.get("text", "") for e in assistant for c in e["message"].get("content", []) if c.get("type") == "text")
    checks = {
        "exit_ok": rc == 0,
        "tools_available_empty": init.get("tools") == [],
        "mcp_servers_empty": init.get("mcp_servers") == [],
        "no_tool_calls": not tool_uses and not tool_results,
        "one_assistant_turn": len(msg_ids) == 1,
        "result_num_turns_1": result.get("num_turns") == 1,
        "cwd_is_temp": "ercot-bench-cc-" in (init.get("cwd") or ""),
    }
    return {"ok": all(checks.values()), "checks": checks, "model": init.get("model"), "text": text,
            "stderr": err[-500:], "cost_usd": result.get("total_cost_usd")}
