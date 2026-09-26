"""ercot-bench CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from ercot_bench.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="ERCOT-Bench: verifiable ERCOT SQL environment")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # gridstatus is very chatty at INFO
    logging.getLogger("gridstatus").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ercot_bench").setLevel(logging.INFO)


ConfigOpt = typer.Option(None, "--config", help="Path to config TOML (default configs/ercot_bench.toml)")


@app.command()
def ingest(
    config: Optional[Path] = ConfigOpt,
    only: Optional[list[str]] = typer.Option(None, help="Only these tables (repeatable), or 'forecasts'"),
    max_age_hours: float = typer.Option(20, help="Re-pull current-year raw files older than this"),
    skip_db: bool = typer.Option(False, help="Only write parquet; don't rebuild ercot.duckdb"),
    verbose: bool = typer.Option(False, "-v"),
):
    """Pull ERCOT data (cached, resumable), write monthly parquet, build data/ercot.duckdb."""
    _setup_logging(verbose)
    from ercot_bench.db import build_database
    from ercot_bench.ingest.build import run_ingest

    cfg = load_config(config)
    summary = run_ingest(cfg, only=only, max_age_hours=max_age_hours)
    for t, months in summary.items():
        typer.echo(f"{t}: wrote {len(months)} month file(s)")
    if not skip_db:
        built = build_database(cfg)
        typer.echo(f"built {cfg.db_path} with tables: {', '.join(built)}")


@app.command("build-db")
def build_db(config: Optional[Path] = ConfigOpt):
    """Rebuild data/ercot.duckdb from existing parquet (no downloads)."""
    _setup_logging(False)
    from ercot_bench.db import build_database

    cfg = load_config(config)
    typer.echo(", ".join(build_database(cfg)))


@app.command()
def schema(config: Optional[Path] = ConfigOpt):
    """Print the schema description shown to models."""
    from ercot_bench.schema_doc import schema_doc

    typer.echo(schema_doc(load_config(config).db_path))


@app.command()
def quality(config: Optional[Path] = ConfigOpt):
    """Data-quality report: rows per month, gaps, duplicates, DST days, null rates."""
    from ercot_bench.quality import quality_report

    typer.echo(quality_report(load_config(config).db_path))


@app.command()
def generate(
    n_per_template: int = typer.Option(50, "--n-per-template", help="Tasks per template per split"),
    config: Optional[Path] = ConfigOpt,
    out_dir: Optional[Path] = typer.Option(None, help="Default: data/tasks"),
    template: Optional[list[str]] = typer.Option(None, help="Only these template ids (repeatable)"),
    paraphrase_model: Optional[str] = typer.Option(None, help="Optional: reword questions with this claude-cli model"),
):
    """Generate tasks with code-computed answers and write train / test_in_template / test_heldout_templates JSONL."""
    _setup_logging(False)
    from ercot_bench.tasks.generator import generate_all, summarize, write_splits
    from ercot_bench.tasks.splits import heldout_template_ids
    from ercot_bench.tasks.templates import ALL_TEMPLATES, TEMPLATES_BY_ID

    cfg = load_config(config)
    tpls = [TEMPLATES_BY_ID[t] for t in template] if template else None
    tasks = generate_all(cfg, n_per_template, tpls)
    if paraphrase_model:
        from ercot_bench.tasks.paraphrase import paraphrase_tasks
        tasks = {s: paraphrase_tasks(ts, paraphrase_model) for s, ts in tasks.items()}
    paths = write_splits(tasks, out_dir or cfg.tasks_dir)
    held = sorted(heldout_template_ids([t.id for t in ALL_TEMPLATES], cfg.seed, cfg.heldout_template_fraction))
    typer.echo(summarize(tasks))
    typer.echo(f"held-out templates: {', '.join(held)}")
    for s, p in paths.items():
        typer.echo(f"wrote {p}")


def _task_file(cfg, split: str) -> Path:
    p = Path(split)
    return p if p.suffix == ".jsonl" else cfg.tasks_dir / f"{split}.jsonl"


@app.command("check-claude-cli")
def check_claude_cli(
    model: str = typer.Option("sonnet", help="Claude model alias or id"),
    split: str = typer.Option("test_in_template", help="Split name or path to a tasks JSONL"),
    config: Optional[Path] = ConfigOpt,
):
    """Run one task through `claude -p --output-format stream-json --verbose` and assert it acted as a plain
    single-turn model: no tools available, no tool calls, exactly one assistant turn."""
    import asyncio

    from ercot_bench.env.prompt import build_prompt
    from ercot_bench.env.score import score
    from ercot_bench.models.claude_cli import check_single_turn
    from ercot_bench.tasks.generator import load_tasks

    cfg = load_config(config)
    task = load_tasks(_task_file(cfg, split))[0]
    system, user = build_prompt(task.question, cfg.db_path)
    res = asyncio.run(check_single_turn(system, user, model))
    for k, v in res["checks"].items():
        typer.echo(f"  [{'PASS' if v else 'FAIL'}] {k}")
    typer.echo(f"model: {res['model']}  cost: {res['cost_usd']}")
    typer.echo(f"question: {task.question}")
    typer.echo(f"response:\n{res['text']}")
    s = score(res["text"], task, cfg.db_path)
    typer.echo(f"score: outcome={s.outcome} reward={s.reward} value={s.value} expected={s.expected}")
    if not res["ok"]:
        typer.echo("CHECK FAILED: do not run CLI-backed evals until this passes.", err=True)
        raise typer.Exit(1)
    typer.echo("CHECK PASSED")


@app.command("eval")
def eval_cmd(
    backend: str = typer.Option(..., help="claude-cli | openai-compat | anthropic-api"),
    model: str = typer.Option(..., help="e.g. sonnet, opus, qwen3:4b, Qwen/Qwen3-8B"),
    split: str = typer.Option("test_in_template", help="Split name or path to tasks JSONL"),
    k: int = typer.Option(10, help="Samples per task"),
    limit: Optional[int] = typer.Option(None, help="Run on a stratified subset of N tasks"),
    concurrency: int = typer.Option(4),
    base_url: Optional[str] = typer.Option(None, help="openai-compat base URL (default $OPENAI_COMPAT_BASE_URL or Ollama)"),
    temperature: float = typer.Option(0.7, help="openai-compat/anthropic-api only; the CLI can't set it"),
    max_tokens: int = typer.Option(4096, help="openai-compat/anthropic-api only"),
    use_n: bool = typer.Option(False, help="openai-compat: request n samples in one call (vLLM)"),
    effort: Optional[str] = typer.Option(None, help="claude-cli: --effort level"),
    no_thinking: bool = typer.Option(False, help="openai-compat: disable Qwen3-style thinking via chat_template_kwargs"),
    out: Optional[Path] = typer.Option(None, help="Output JSONL (default results/<split>__<backend>__<model>.jsonl)"),
    config: Optional[Path] = ConfigOpt,
):
    """Run a model over a split (k samples/task). Resumable: rerun to continue."""
    import asyncio

    _setup_logging(False)
    from ercot_bench.config import REPO_ROOT
    from ercot_bench.eval.run import default_output_path, run_eval, stratified_subset
    from ercot_bench.models.base import make_client
    from ercot_bench.tasks.generator import load_tasks

    cfg = load_config(config)
    tasks = stratified_subset(load_tasks(_task_file(cfg, split)), limit)
    kw = {"base_url": base_url, "temperature": temperature, "max_tokens": max_tokens, "use_n": use_n, "effort": effort,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": False}} if no_thinking else None}
    client = make_client(backend, model, **{k_: v for k_, v in kw.items() if v is not None})
    split_name = Path(split).stem
    out = out or default_output_path(REPO_ROOT / "results", split_name, backend, model)
    stats = asyncio.run(run_eval(client, tasks, cfg.db_path, out, k=k, concurrency=concurrency,
                                 timeout_s=cfg.query_timeout_s))
    typer.echo(json.dumps(stats))
    typer.echo(f"results: {out}")
    if stats["rate_limited"]:
        typer.echo("Stopped on usage/rate limits. Rerun the same command later to resume.", err=True)
        raise typer.Exit(2)


@app.command()
def report(
    results: list[Path] = typer.Argument(None, help="Result JSONL files (default: all in results/)"),
    out_dir: Path = typer.Option(Path("results/report"), help="Where to write report.md and report.csv"),
):
    """Aggregate results: pass@1/5/10, consistency, never-solved, breakdowns, outcomes, latency/cost."""
    from ercot_bench.config import REPO_ROOT
    from ercot_bench.eval.report import build_report, load_results

    files = results or sorted((REPO_ROOT / "results").glob("*.jsonl"))
    if not files:
        typer.echo("no result files", err=True)
        raise typer.Exit(1)
    typer.echo(build_report(load_results(files), out_dir))
    typer.echo(f"\nwrote {out_dir}/report.md and report.csv")


@app.command("sft-build")
def sft_build(
    results: list[Path] = typer.Argument(..., help="Result JSONL(s) from a frontier model on the train split"),
    out: Path = typer.Option(Path("data/sft/train.jsonl"), help="Keep alone in its dir: prime-rl loads the dir"),
    max_per_task: int = typer.Option(2, help="Cap examples per task after dedupe"),
    tasks_file: Optional[Path] = typer.Option(None, help="Tasks JSONL (default data/tasks/train.jsonl)"),
    config: Optional[Path] = ConfigOpt,
):
    """Build a chat-format SFT dataset from correct completions (deduped by SQL, capped per task)."""
    from ercot_bench.sft.build import build_sft

    cfg = load_config(config)
    summary = build_sft(results, tasks_file or cfg.tasks_dir / "train.jsonl", cfg.db_path, out, max_per_task)
    typer.echo(summary)


@app.command("rl-subset")
def rl_subset(
    results: list[Path] = typer.Argument(..., help="Result JSONL(s) of the SFT (or base) model on the train split"),
    out: Path = typer.Option(Path("data/tasks/rl_train.jsonl")),
    tasks_file: Optional[Path] = typer.Option(None, help="Tasks JSONL (default data/tasks/train.jsonl)"),
    solved_keep_frac: float = typer.Option(0.1, help="Fraction of always-solved tasks to keep"),
    config: Optional[Path] = ConfigOpt,
):
    """Build the RL prompt set weighted toward tasks the model gets wrong or inconsistently right.

    Tasks with 0 < pass rate < 1 (most GRPO signal) are written 3x, never-solved 1x, always-solved are
    subsampled. Tasks not in the results are kept 1x."""
    import random
    from collections import defaultdict

    from ercot_bench.tasks.generator import load_tasks

    cfg = load_config(config)
    tasks = load_tasks(tasks_file or cfg.tasks_dir / "train.jsonl")
    stats: dict[str, list[int]] = defaultdict(list)
    for f in results:
        for line in open(f):
            r = json.loads(line)
            stats[r["task_id"]].append(int(r["outcome"] == "correct"))
    rng = random.Random(cfg.seed)
    counts = {"mixed": 0, "never": 0, "always_kept": 0, "always_dropped": 0, "unseen": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fo:
        for t in tasks:
            s = stats.get(t.task_id)
            if not s:
                reps, k = 1, "unseen"
            elif 0 < sum(s) < len(s):
                reps, k = 3, "mixed"
            elif sum(s) == 0:
                reps, k = 1, "never"
            elif rng.random() < solved_keep_frac:
                reps, k = 1, "always_kept"
            else:
                reps, k = 0, "always_dropped"
            counts[k] += 1
            for _ in range(reps):
                fo.write(t.model_dump_json() + "\n")
    typer.echo(f"wrote {out}: {counts}")


if __name__ == "__main__":
    app()
