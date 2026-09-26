"""Check that rewards recorded by a verifiers `eval` run equal our own harness's score() on the same replies.

    uv run python scripts/compare_verifiers_rewards.py <verifiers output dir or traces.jsonl> [...]

Walks every traces.jsonl, finds each trace (an object with task/nodes/rewards), takes the last sampled
assistant reply, re-scores it with ercot_bench.env.score.score, and compares to the traced reward.
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ercot_bench.config import load_config
from ercot_bench.env.score import score
from ercot_bench.tasks.schema import Task


def _traces(obj):
    if isinstance(obj, dict):
        if {"task", "nodes", "rewards"} <= obj.keys():
            yield obj
        for v in obj.values():
            yield from _traces(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _traces(v)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _last_reply(trace: dict) -> str:
    replies = [n["message"] for n in trace["nodes"]
               if n.get("sampled") and (n.get("message") or {}).get("role") == "assistant"]
    return _text(replies[-1].get("content")).strip() if replies else ""


def _bench_task(trace: dict) -> Task | None:
    for d in _traces_data(trace["task"]):
        return Task.model_validate_json(d)
    return None


def _traces_data(obj):
    if isinstance(obj, dict):
        if "bench_task" in obj:
            yield obj["bench_task"]
        for v in obj.values():
            yield from _traces_data(v)


def _lines(f: Path):
    """traces.jsonl, or the zstd-compressed trace streams newer verifiers write under monitors/file/traces/."""
    if f.suffix == ".zst":
        import io

        import zstandard

        with f.open("rb") as fh:
            yield from io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
    else:
        yield from f.open()


def main(paths: list[str]) -> int:
    db = load_config().db_path
    files = []
    for p in map(Path, paths):
        files += (sorted(p.rglob("traces.jsonl")) + sorted(p.rglob("*.jsonl.zst"))) if p.is_dir() else [p]
    n = bad = 0
    for f in files:
        for line in _lines(f):
            episode = json.loads(line)
            for tr in _traces(episode):
                # the task data (incl. bench_task) sits on the trace in traces.jsonl, or on the episode record
                task = _bench_task(tr) or _bench_task(episode)
                reward = (tr["rewards"].get("ercot_sql_reward") or {})
                traced = (reward.get("score", reward.get("value")) if isinstance(reward, dict) else reward)
                if task is None or traced is None:
                    continue
                ours = score(_last_reply(tr), task, db)
                n += 1
                if abs(ours.reward - float(traced)) > 1e-9:
                    bad += 1
                    print(f"MISMATCH {task.task_id}: verifiers={traced} harness={ours.reward} ({ours.outcome})")
    print(f"compared {n} rollouts from {len(files)} file(s): {n - bad} match, {bad} mismatch")
    return 1 if bad or n == 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
