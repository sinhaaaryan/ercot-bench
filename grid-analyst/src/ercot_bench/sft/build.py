"""Build a chat-format SFT dataset from correct frontier-model completions."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ercot_bench.env.prompt import build_prompt
from ercot_bench.env.score import extract_sql
from ercot_bench.tasks.generator import load_tasks


def _norm_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def build_sft(result_files: list[Path], tasks_file: Path, db_path: Path, out: Path, max_per_task: int = 2) -> str:
    tasks = {t.task_id: t for t in load_tasks(tasks_file)}
    per_task: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    attempted: set[str] = set()
    for f in result_files:
        for line in open(f):
            r = json.loads(line)
            if r["task_id"] not in tasks:
                continue
            attempted.add(r["task_id"])
            if r["outcome"] != "correct":
                continue
            sql = extract_sql(r["completion"])
            key = _norm_sql(sql or "")
            if not sql or key in seen[r["task_id"]] or len(per_task[r["task_id"]]) >= max_per_task:
                continue
            seen[r["task_id"]].add(key)
            per_task[r["task_id"]].append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    by_template: Counter = Counter()
    with out.open("w") as fo:
        for tid in sorted(per_task):
            task = tasks[tid]
            system, user = build_prompt(task.question, db_path)
            for r in per_task[tid]:
                fo.write(json.dumps({
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                 {"role": "assistant", "content": r["completion"].strip()}],
                    "task_id": tid, "template_id": task.template_id, "source_model": r["model"],
                }) + "\n")
                n += 1
                by_template[task.template_id] += 1
    zero = sorted(attempted - set(per_task))
    lines = [f"wrote {n} examples to {out}",
             f"tasks attempted: {len(attempted)}, covered (>=1 correct): {len(per_task)}, zero correct: {len(zero)}",
             "examples per template: " + ", ".join(f"{k}={v}" for k, v in sorted(by_template.items()))]
    # keep side files out of the dataset dir: prime-rl's `sft` loads the whole directory with load_dataset()
    zero_path = out.parent.parent / "sft_meta" / "zero_correct_task_ids.txt"
    zero_path.parent.mkdir(parents=True, exist_ok=True)
    zero_path.write_text("\n".join(zero) + ("\n" if zero else ""))
    lines.append(f"zero-correct task ids (good RL targets): {zero_path}")
    return "\n".join(lines)
