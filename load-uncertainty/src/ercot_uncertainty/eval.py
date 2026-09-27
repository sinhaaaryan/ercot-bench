"""Evaluate predictors on data/examples/<split>.jsonl with the same scoring as RL training.

Backends
  baseline      echo the example's numeric-baseline percentiles (reward 0 by construction)
  openai        any OpenAI-compatible server, e.g. local vLLM serving the base or RL model
  claude-cli    frontier Claude via Claude Code headless mode (`claude -p`, logged-in subscription)

Results go to results/<split>__<backend>__<model>.jsonl (resumable; rerun to continue), then
`python -m ercot_uncertainty.eval report` prints the leaderboard.

  uv run python -m ercot_uncertainty.eval run --backend claude-cli --model sonnet --split test_sample400
  uv run python -m ercot_uncertainty.eval run --backend openai --model Qwen/Qwen3-4B-Instruct-2507 --split test_sample400
  uv run python -m ercot_uncertainty.eval report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ercot_uncertainty.baselines.scoring import coverage, mean_quantile_loss
from ercot_uncertainty.task import parse_percentiles, reward_percentiles

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "data" / "examples"
RESULTS = Path(os.environ.get("ERCOT_RESULTS_DIR", ROOT / "results"))  # override for pipeline tests


def load_examples(split: str) -> list[dict]:
    path = Path(split) if split.endswith(".jsonl") else EXAMPLES / f"{split}.jsonl"
    return [json.loads(line) for line in path.open()]


def _int_keys(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


def baseline_json(ex: dict) -> str:
    base = _int_keys(ex["baseline"])
    return json.dumps({"hours": [{"he": h, **base[h]} for h in sorted(base)]})


@dataclass
class Scored:
    id: str
    completion: str
    reward: float
    parse_ok: bool
    pinball_mw: float | None
    baseline_pinball_mw: float
    covered: float | None
    width_mw: float | None
    latency_s: float | None = None
    cost_usd: float | None = None
    output_tokens: int | None = None
    served_model: str | None = None  # model that actually answered (claude-cli can silently remap retired ids)
    thinking_tokens: int | None = None
    error: str | None = None


def score(ex: dict, completion: str, **extra: Any) -> Scored:
    base, actual = _int_keys(ex["baseline"]), _int_keys(ex["actual_error"])
    hours = sorted(actual)
    ys = [actual[h] for h in hours]
    parsed = parse_percentiles(completion, hours)
    reward = reward_percentiles(completion, base, actual)
    base_loss = mean_quantile_loss([base[h] for h in hours], ys)
    if parsed.ok:
        preds = [parsed.hours[h] for h in hours]
        return Scored(ex["id"], completion, reward, True, mean_quantile_loss(preds, ys), base_loss,
                      coverage(preds, ys), float(np.mean([p["p90"] - p["p10"] for p in preds])), **extra)
    return Scored(ex["id"], completion, reward, False, None, base_loss, None, None, **extra)


# --- backends ------------------------------------------------------------------------------


class Backend:
    name = "base"

    async def complete(self, messages: list[dict]) -> dict:  # -> {"text", "latency_s", ...}
        raise NotImplementedError


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, model: str, base_url: str, temperature: float = 0.0, max_tokens: int = 512):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    async def complete(self, messages):
        t0 = time.time()
        r = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        return {"text": r.choices[0].message.content or "", "latency_s": time.time() - t0,
                "output_tokens": r.usage.completion_tokens if r.usage else None}


class ClaudeCLIBackend(Backend):
    """Claude Code as a plain single-turn model: no tools, no MCP, no settings, empty temp cwd.

    Flags follow the teammate's verified harness (/hackathon/ercot-bench/.../claude_cli.py).
    Residue we cannot remove without an API key: the CLI prepends a short identity line and an
    environment block (incl. today's date) to the system prompt.
    """

    name = "claude-cli"
    RATE = ("usage limit", "rate limit", "rate_limit", "limit reached", "429", "overloaded", "529")

    def __init__(self, model: str, effort: str | None = None, thinking: bool = False, timeout_s: float = 300):
        self.model, self.effort, self.thinking, self.timeout_s = model, effort, thinking, timeout_s

    def _args(self, system: str) -> list[str]:
        args = [shutil.which("claude") or "claude", "-p", "--system-prompt", system, "--tools", "",
                "--max-turns", "1", "--strict-mcp-config", "--setting-sources", "",
                "--disable-slash-commands", "--no-session-persistence", "--model", self.model,
                "--output-format", "json"]
        if self.effort:
            args += ["--effort", self.effort]
        if not self.thinking:
            args += ["--settings", '{"alwaysThinkingEnabled": false}']
        return args

    async def complete(self, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        for attempt in range(5):
            t0 = time.time()
            with tempfile.TemporaryDirectory(prefix="ercot-unc-cc-") as cwd:
                proc = await asyncio.create_subprocess_exec(
                    *self._args(system), cwd=cwd, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                try:
                    out, err = await asyncio.wait_for(proc.communicate(user.encode()), self.timeout_s)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    out, err = b"", b"timeout"
            try:
                data = json.loads(out) if out.strip() else {}
            except json.JSONDecodeError:
                data = {}
            if data and not data.get("is_error") and proc.returncode == 0:
                usage = data.get("usage") or {}
                served = ",".join(sorted((data.get("modelUsage") or {}).keys()))
                if served and self.model not in served:
                    print(f"WARNING: requested {self.model} but claude-cli served {served}", flush=True)
                return {"text": data.get("result") or "", "latency_s": time.time() - t0,
                        "cost_usd": data.get("total_cost_usd"), "output_tokens": usage.get("output_tokens"),
                        "served_model": served,
                        "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens")}
            msg = (data.get("result") or err.decode(errors="replace") or out.decode(errors="replace"))[:300]
            if "not logged in" in msg.lower():
                raise RuntimeError(f"claude CLI not logged in: {msg}")
            wait = 30 * 2**attempt if any(m in msg.lower() for m in self.RATE) else 3
            await asyncio.sleep(wait)
        return {"text": "", "latency_s": None, "error": msg}


class BaselineBackend(Backend):
    name = "baseline"


def make_backend(args) -> Backend:
    if args.backend == "openai":
        return OpenAIBackend(args.model, args.base_url, args.temperature, args.max_tokens)
    if args.backend == "claude-cli":
        return ClaudeCLIBackend(args.model, args.effort, args.thinking)
    if args.backend == "baseline":
        return BaselineBackend()
    raise ValueError(args.backend)


# --- run / report --------------------------------------------------------------------------


def out_path(split: str, backend: str, model: str) -> Path:
    p = Path(split)
    variant = p.parent.name if p.parent.name in ("numeric", "text", "textswap", "textswap_text") else ""
    stem = f"{p.stem}-{variant}" if variant else p.stem  # numeric/text prompt variants get their own files
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{stem}__{backend}__{model}")
    return RESULTS / f"{tag}.jsonl"


async def run(args) -> Path:
    examples = load_examples(args.split)[: args.limit or None]
    backend = make_backend(args)
    tag = (args.model or "numeric") + ("+thinking" if getattr(args, "thinking", False) else "") + \
        (f"+effort-{args.effort}" if getattr(args, "effort", None) else "")
    out = out_path(args.split, backend.name, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    todo = [ex for ex in examples if ex["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go -> {out}")
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def one(ex):
        async with sem:
            if isinstance(backend, BaselineBackend):
                r = {"text": baseline_json(ex), "latency_s": 0.0}
            else:
                r = await backend.complete(ex["prompt"])
            s = score(ex, r.pop("text"), **r)
            async with lock:
                with out.open("a") as f:
                    f.write(json.dumps(asdict(s)) + "\n")

    await asyncio.gather(*(one(ex) for ex in todo))
    return out


def rescore(path: Path, split: str) -> None:
    """Recompute every score in a results file from its saved completions (after a parser/reward change)."""
    ex = {e["id"]: e for e in load_examples(split)}
    rows = [json.loads(line) for line in path.open()]
    keep = ("latency_s", "cost_usd", "output_tokens", "served_model", "thinking_tokens", "error")
    out = [asdict(score(ex[r["id"]], r["completion"], **{k: r.get(k) for k in keep})) for r in rows]
    before = sum(not r["parse_ok"] for r in rows)
    after = sum(not r["parse_ok"] for r in out)
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{path.name}: {len(out)} rows rescored, parse failures {before} -> {after}")


def summarize(path: Path, examples: dict[str, dict] | None = None) -> dict:
    df = pd.read_json(path, lines=True)
    ok = df[df["parse_ok"]]
    row = {
        "run": path.stem,
        "n": len(df),
        "parse_fail_%": 100 * (1 - df["parse_ok"].mean()),
        "reward": df["reward"].mean(),
        "pinball_MW": ok["pinball_mw"].mean(),
        "baseline_pinball_MW": df["baseline_pinball_mw"].mean(),
        "coverage_80": ok["covered"].mean(),
        "width_MW": ok["width_mw"].mean(),
        "latency_s": df["latency_s"].mean() if "latency_s" in df else None,
        "cost_usd_per_pred": df["cost_usd"].mean() if "cost_usd" in df else None,
    }
    if examples:
        agreed = df["id"].map(lambda i: (examples.get(i, {}).get("meta") or {}).get("models_agreed"))
        if agreed.notna().any():
            sub = df[agreed.fillna(False).astype(bool)]
            row["reward_models_agreed"] = sub["reward"].mean()
    return row


def compare(files: list[Path], split: str, first_n: int | None = None, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Paired comparison vs the numeric baseline on the examples every run has (optionally the split's first N).

    Zone-days on the same operating day share weather and forecaster text, so they are not independent:
    confidence intervals come from a bootstrap over operating DAYS (all zones of a resampled day move together).
    delta_MW < 0 means lower pinball loss than the reference (the first file) = better.
    """
    ex = {e["id"]: e for e in load_examples(split)}
    order = list(ex)[:first_n] if first_n else list(ex)
    runs = {f.stem: {json.loads(l)["id"]: json.loads(l) for l in f.open()} for f in files}
    ids = [i for i in order if all(i in r for r in runs.values())]
    days = np.array([ex[i]["operating_date"] for i in ids])
    uniq = np.unique(days)
    idx_by_day = [np.where(days == d)[0] for d in uniq]
    # reference = the FIRST results file given (its own pinball loss), so deltas never depend on which baseline
    # version a run's prompts happened to embed
    ref = next(iter(runs.values()))
    fallback = lambda i: np.mean(np.abs(list(ex[i]["actual_error"].values()))) * 0.5  # "trust ERCOT exactly" (p10=p50=p90=0)
    base = np.array([ref[i]["pinball_mw"] if ref[i]["pinball_mw"] is not None else fallback(i) for i in ids])
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([idx_by_day[j] for j in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]
    rows = []
    for name, r in runs.items():
        # unparseable answers score the worst clipped reward; for MW use the loss of a zero-width forecast at 0
        loss = np.array([r[i]["pinball_mw"] if r[i]["pinball_mw"] is not None else
                         np.mean(np.abs(list(ex[i]["actual_error"].values()))) * 0.5 for i in ids])
        d = loss - base
        bd = np.array([d[b].mean() for b in boots])
        cov = np.array([r[i]["covered"] if r[i]["covered"] is not None else 0.0 for i in ids])
        rows.append({"run": name, "n": len(ids), "days": len(uniq), "pinball_MW": loss.mean(), "baseline_MW": base.mean(),
                     "delta_MW": d.mean(), "delta_%": 100 * d.mean() / base.mean(),
                     "ci95_%": f"[{100 * np.percentile(bd, 2.5) / base.mean():+.1f}, {100 * np.percentile(bd, 97.5) / base.mean():+.1f}]",
                     "p_better": float(np.mean(bd < 0)), "coverage_80": cov.mean(),
                     "parse_fail": sum(not r[i]["parse_ok"] for i in ids)})
    return pd.DataFrame(rows).sort_values("delta_MW")


def report(split: str | None = None) -> pd.DataFrame:
    files = sorted(RESULTS.glob(f"{split}__*.jsonl" if split else "*.jsonl"))
    examples = {}
    if split and (EXAMPLES / f"{split}.jsonl").exists():
        examples = {ex["id"]: ex for ex in load_examples(split)}
    df = pd.DataFrame([summarize(f, examples) for f in files])
    if not df.empty:
        df = df.sort_values("reward", ascending=False)
    return df


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["baseline", "openai", "claude-cli"], required=True)
    r.add_argument("--model", default="")
    r.add_argument("--split", default="test")
    r.add_argument("--limit", type=int, default=0, help="first N rows (files are zone-ordered: use val_sample200 / test_sample400 for balanced subsets)")
    r.add_argument("--base-url", default="http://localhost:8300/v1", help="our vLLM router (8000/8001/8003 are the teammate's)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--effort", default=None)
    r.add_argument("--thinking", action="store_true", help="claude-cli: allow extended thinking")
    r.add_argument("--concurrency", type=int, default=4)
    rs = sub.add_parser("rescore")
    rs.add_argument("results", nargs="+")
    rs.add_argument("--split", required=True)
    cp = sub.add_parser("compare", help="paired, day-clustered bootstrap comparison vs the baseline")
    cp.add_argument("results", nargs="+")
    cp.add_argument("--split", required=True)
    cp.add_argument("--first-n", type=int, default=None)
    rep = sub.add_parser("report")
    rep.add_argument("--split", default=None)
    a = p.parse_args()
    if a.cmd == "run":
        print(asyncio.run(run(a)))
    elif a.cmd == "compare":
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(compare([Path(f) for f in a.results], a.split, a.first_n).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    elif a.cmd == "rescore":
        for f in a.results:
            rescore(Path(f), a.split)
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(report(a.split).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
