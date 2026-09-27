"""Aggregate eval JSONL results into markdown + CSV reports."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from statistics import mean

OUTCOMES = ("correct", "wrong_answer", "sql_error", "format_error")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_results(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def _metrics(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    ns = {t: len(v) for t, v in by_task.items()}
    cs = {t: sum(r["outcome"] == "correct" for r in v) for t, v in by_task.items()}
    k_min = min(ns.values()) if ns else 0
    m = {"tasks": len(by_task), "samples": len(rows), "k_min": k_min,
         "pass@1": mean(cs[t] / ns[t] for t in by_task) if by_task else 0.0}
    for k in (5, 10):
        m[f"pass@{k}"] = mean(pass_at_k(ns[t], cs[t], k) for t in by_task) if by_task and k_min >= k else None
    m["consistency"] = float(mean(cs[t] == ns[t] for t in by_task)) if by_task else 0.0
    m["never_solved"] = sum(cs[t] == 0 for t in by_task)
    oc = Counter(r["outcome"] for r in rows)
    for o in OUTCOMES:
        m[o] = oc.get(o, 0) / len(rows) if rows else 0.0
    m["mean_reward"] = mean(r["reward"] for r in rows) if rows else 0.0
    m["mean_latency_s"] = mean(r["latency_s"] for r in rows) if rows else None
    out_toks = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
    m["mean_output_tokens"] = mean(out_toks) if out_toks else None
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    m["total_cost_usd"] = sum(costs) if costs else None
    return m


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _table(title: str, rows: list[tuple[str, dict]], cols: list[str]) -> str:
    lines = [f"### {title}", "", "| group | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, m in rows:
        lines.append(f"| {name} | " + " | ".join(_fmt(m.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def build_report(rows: list[dict], out_dir: Path | None = None) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["backend"], r["split"])].append(r)
    main_cols = ["tasks", "samples", "k_min", "pass@1", "pass@5", "pass@10", "consistency", "never_solved",
                 "correct", "wrong_answer", "sql_error", "format_error", "mean_reward", "mean_latency_s",
                 "mean_output_tokens", "total_cost_usd"]
    md = ["# ERCOT-Bench report", ""]
    csv_rows = []
    overall = []
    for (model, backend, split), rs in sorted(groups.items()):
        m = _metrics(rs)
        overall.append((f"{model} ({backend}) / {split}", m))
        csv_rows.append({"model": model, "backend": backend, "split": split, "group_type": "overall", "group": "all", **m})
    md.append(_table("Overall", overall, main_cols))
    sub_cols = ["tasks", "samples", "pass@1", "pass@5", "consistency", "never_solved", "sql_error", "format_error"]
    for key, label in (("family", "family"), ("difficulty", "difficulty"), ("template_id", "template")):
        for (model, backend, split), rs in sorted(groups.items()):
            by: dict[str, list[dict]] = defaultdict(list)
            for r in rs:
                by[str(r[key])].append(r)
            sub = [(g, _metrics(v)) for g, v in sorted(by.items())]
            md.append(_table(f"By {label}: {model} ({backend}) / {split}", sub, sub_cols))
            for g, m in sub:
                csv_rows.append({"model": model, "backend": backend, "split": split, "group_type": label, "group": g, **m})
    text = "\n".join(md)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(text)
        fields = ["model", "backend", "split", "group_type", "group"] + main_cols
        with open(out_dir / "report.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)
    return text
