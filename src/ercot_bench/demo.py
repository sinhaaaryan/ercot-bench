"""Interactive demo: ask one or more models an ERCOT question, run their SQL, and grade it when ground truth exists.

    ercot-bench ask                      # REPL comparing sft, base, haiku side by side
    ercot-bench ask --targets sft        # one model
    ercot-bench ask --example 3          # run curated example 3 and exit

Targets are defined in TARGETS below (override URLs/models with flags).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ercot_bench.config import REPO_ROOT, load_config
from ercot_bench.env.execute import execute_sql, jsonable
from ercot_bench.env.prompt import build_prompt
from ercot_bench.env.score import extract_sql, score_exec
from ercot_bench.models.base import make_client
from ercot_bench.tasks.generator import load_tasks
from ercot_bench.tasks.schema import Task

SFT_WEIGHTS = "/hackathon/outputs/ercot-sft-1p7b/checkpoints/step_80/weights"
SFT8B_WEIGHTS = "/hackathon/outputs/ercot-sft-8b-lora/export/merged"
K2SFT_WEIGHTS = "/hackathon/outputs/ercot-sft-k2-lora/merged"


@dataclass
class Target:
    label: str
    backend: str
    model: str
    kwargs: dict


def default_targets(sft_url: str, base_url: str, sft_model: str, sft8b_url: str = "http://localhost:8002/v1",
                    sft8b_model: str = SFT8B_WEIGHTS) -> dict[str, Target]:
    qwen = {"temperature": 0.0, "max_tokens": 1536, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {
        "sft": Target("Qwen3-1.7B + SFT", "openai-compat", sft_model, {"base_url": sft_url, **qwen}),
        "base": Target("Qwen3-1.7B (base)", "openai-compat", "Qwen/Qwen3-1.7B", {"base_url": base_url, **qwen}),
        "sft8b": Target("Qwen3-8B + SFT", "openai-compat", sft8b_model, {"base_url": sft8b_url, **qwen}),
        "k2sft": Target("K2-Horizon-7B + SFT", "openai-compat", K2SFT_WEIGHTS,
                        {"base_url": "http://localhost:8003/v1", "temperature": 0.0, "max_tokens": 2048,
                         "extra_body": {"chat_template_kwargs": {"reasoning_effort": "low"}}}),
        "haiku": Target("Claude Haiku 4.5", "claude-cli", "haiku", {}),
        "sonnet": Target("Claude Sonnet", "claude-cli", "sonnet", {}),
    }


def load_examples() -> dict:
    return json.loads((REPO_ROOT / "configs" / "demo_examples.json").read_text())


def all_test_tasks(cfg) -> dict[str, Task]:
    out = {}
    for split in ("test_in_template", "test_heldout_templates", "train"):
        p = cfg.tasks_dir / f"{split}.jsonl"
        if p.exists():
            out.update({t.task_id: t for t in load_tasks(p)})
    return out


async def _run_target(t: Target, system: str, user: str, db_path: Path, task: Task | None) -> dict:
    client = make_client(t.backend, t.model, **t.kwargs)
    t0 = time.time()
    try:
        comp = (await client.generate(system, user, n=1))[0]
    except Exception as e:  # noqa: BLE001
        return {"target": t, "error": f"{type(e).__name__}: {e}", "latency": time.time() - t0}
    out = {"target": t, "latency": time.time() - t0, "text": comp.text, "error": comp.error}
    sql = extract_sql(comp.text)
    out["sql"] = sql
    if sql is None:
        out["verdict"] = "format_error (no ```sql block)" if task else "no SQL"
        return out
    res = await asyncio.to_thread(execute_sql, sql, db_path)
    out["exec"] = res
    if task is not None:
        s = score_exec(sql, res, task)
        out["verdict"] = s.outcome
        out["value"] = s.value
    return out


def _render(console: Console, question: str, task: Task | None, results: list[dict]) -> None:
    title = f"[bold]{task.template_id}[/] ({task.family}, difficulty {task.difficulty}, {task.split})" if task else "free-form"
    console.print(Panel(question, title=title, border_style="cyan"))
    if task is not None:
        console.print(f"[bold]Ground truth:[/] {task.answer!r}  [dim](computed by our reference code)[/]")
    summary = Table(show_header=True, header_style="bold")
    for col in ("model", "result", "verdict", "latency"):
        summary.add_column(col)
    for r in results:
        t = r["target"]
        console.rule(f"[bold]{t.label}[/]")
        if r.get("error"):
            console.print(f"[red]backend error:[/] {r['error']}")
            summary.add_row(t.label, "-", "[red]error[/]", f"{r['latency']:.1f}s")
            continue
        if r.get("sql"):
            console.print(Syntax(r["sql"], "sql", word_wrap=True, theme="ansi_dark"))
        else:
            console.print(r.get("text", "")[:800])
        res = r.get("exec")
        if res is None:
            shown = "-"
        elif not res.ok:
            shown = f"SQL error: {res.error[:120]}"
        else:
            rows = [jsonable(list(x)) for x in res.rows[:6]]
            shown = json.dumps(rows[0][0] if len(rows) == 1 and len(rows[0]) == 1 else rows, default=str)[:200]
        verdict = r.get("verdict", "(not graded)")
        color = {"correct": "green", "wrong_answer": "red", "sql_error": "red"}.get(verdict, "yellow")
        console.print(f"result: {shown}")
        summary.add_row(t.label, shown, f"[{color}]{verdict}[/]", f"{r['latency']:.1f}s")
    console.rule("summary")
    console.print(summary)


def ask_once(question: str, task: Task | None, targets: list[Target], console: Console) -> None:
    cfg = load_config()
    system, user = build_prompt(question, cfg.db_path)

    async def main():
        return await asyncio.gather(*(_run_target(t, system, user, cfg.db_path, task) for t in targets))

    with console.status(f"asking {', '.join(t.label for t in targets)} ..."):
        results = asyncio.run(main())
    _render(console, question, task, results)


def repl(targets: list[Target], console: Console) -> None:
    cfg = load_config()
    ex = load_examples()
    tasks = all_test_tasks(cfg)
    help_text = ("[bold]Commands:[/] [cyan]list[/] (curated examples) · [cyan]<n>[/] run example n · "
                 "[cyan]f<n>[/] free-form example n · [cyan]random[/] \\[family] (random test task) · "
                 "any other text = your own question · [cyan]quit[/]")
    console.print(Panel(f"Models: {', '.join(t.label for t in targets)}\n{help_text}", title="ERCOT-Bench demo",
                        border_style="green"))
    while True:
        try:
            line = console.input("[bold green]ercot>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line in ("help", "?"):
            console.print(help_text)
            continue
        if line == "list":
            tbl = Table("#", "example", "recorded result")
            for i, e in enumerate(ex["examples"], 1):
                tbl.add_row(str(i), e["title"], e.get("note", ""))
            for i, q in enumerate(ex["freeform"], 1):
                tbl.add_row(f"f{i}", q[:90] + ("..." if len(q) > 90 else ""), "free-form (not graded)")
            console.print(tbl)
            continue
        task = None
        if line.isdigit() and 1 <= int(line) <= len(ex["examples"]):
            task = tasks.get(ex["examples"][int(line) - 1]["task_id"])
            if task is None:
                console.print("[red]task not found; run `ercot-bench generate` first[/]")
                continue
            question = task.question
        elif line.startswith("f") and line[1:].isdigit() and 1 <= int(line[1:]) <= len(ex["freeform"]):
            question = ex["freeform"][int(line[1:]) - 1]
        elif line.split()[0] == "random":
            fam = line.split()[1] if len(line.split()) > 1 else None
            pool = [t for t in tasks.values() if t.split != "train" and (fam is None or t.family == fam)]
            if not pool:
                console.print(f"[red]no test tasks for family {fam!r}[/]")
                continue
            task = random.choice(pool)
            question = task.question
        else:
            question = line
        ask_once(question, task, targets, console)
