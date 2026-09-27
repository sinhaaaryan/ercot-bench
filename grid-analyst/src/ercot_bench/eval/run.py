"""Run a model over a task split with k samples per task. Resumable, concurrent, JSONL output."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from ercot_bench.env.prompt import build_prompt
from ercot_bench.env.score import score
from ercot_bench.models.base import BackendUnavailable, ModelClient, RateLimited
from ercot_bench.tasks.schema import Task

log = logging.getLogger("ercot_bench.eval")


def default_output_path(results_dir: Path, split: str, backend: str, model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    return results_dir / f"{split}__{backend}__{safe}.jsonl"


def stratified_subset(tasks: list[Task], limit: int | None) -> list[Task]:
    """Round-robin across templates so a small --limit still covers every template."""
    if not limit or limit >= len(tasks):
        return sorted(tasks, key=lambda t: t.task_id)
    by_t: dict[str, list[Task]] = defaultdict(list)
    for t in sorted(tasks, key=lambda t: t.task_id):
        by_t[t.template_id].append(t)
    out: list[Task] = []
    i = 0
    while len(out) < limit:
        added = False
        for tid in sorted(by_t):
            if i < len(by_t[tid]) and len(out) < limit:
                out.append(by_t[tid][i])
                added = True
        if not added:
            break
        i += 1
    return out


def _done_pairs(path: Path) -> set[tuple[str, int]]:
    done = set()
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["sample_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def run_eval(client: ModelClient, tasks: list[Task], db_path: Path, out_path: Path, k: int = 10,
                   concurrency: int = 4, timeout_s: float = 10.0, progress: bool = True) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_pairs(out_path)
    jobs = [(t, i) for t in tasks for i in range(k) if (t.task_id, i) not in done]
    log.info("%d tasks x k=%d: %d samples to run (%d already done) -> %s", len(tasks), k, len(jobs), len(done), out_path)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    stop = asyncio.Event()
    stats = {"written": 0, "backend_errors": 0, "rate_limited": False, "remaining": len(jobs)}
    desc = client.describe()

    async def one(task: Task, idx: int):
        if stop.is_set():
            return
        async with sem:
            if stop.is_set():
                return
            system, user = build_prompt(task.question, db_path)
            try:
                comp = (await client.generate(system, user, n=1))[0]
            except (RateLimited, BackendUnavailable) as e:
                stats["rate_limited"] = True
                if not stop.is_set():
                    log.error("%s: %s -- stopping; rerun the same command later to resume", type(e).__name__, e)
                stop.set()
                return
            if comp.error:
                stats["backend_errors"] += 1
                log.warning("backend error on %s#%d (not recorded, will retry on rerun): %s", task.task_id, idx, comp.error[:200])
                return
            res = await asyncio.to_thread(score, comp.text, task, db_path, timeout_s)
            rec = {
                "task_id": task.task_id, "template_id": task.template_id, "family": task.family,
                "difficulty": task.difficulty, "split": task.split, "sample_idx": idx,
                "backend": desc["backend"], "model": desc["model"], "backend_meta": desc,
                "completion": comp.text, "sql": res.sql, "value": res.value, "expected": res.expected,
                "outcome": res.outcome, "reward": res.reward, "detail": res.detail,
                "input_tokens": comp.input_tokens, "output_tokens": comp.output_tokens, "cost_usd": comp.cost_usd,
                "latency_s": round(comp.latency_s, 3), "completion_meta": comp.meta,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            async with write_lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                stats["written"] += 1
                if progress and stats["written"] % 10 == 0:
                    log.info("progress: %d/%d samples", stats["written"], len(jobs))

    await asyncio.gather(*(one(t, i) for t, i in jobs))
    stats["remaining"] = len(jobs) - stats["written"]
    return stats
