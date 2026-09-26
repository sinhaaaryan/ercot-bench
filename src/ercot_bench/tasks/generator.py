"""Generate tasks: sample params per template, compute ground truth, assign splits, write JSONL."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

from ercot_bench.config import Config
from ercot_bench.db import connect_readonly
from ercot_bench.tasks.schema import Task, Tolerance
from ercot_bench.tasks.splits import SPLITS, TEST_HELDOUT, TRAIN, heldout_template_ids, split_years
from ercot_bench.tasks.templates import ALL_TEMPLATES, Ctx, Template

log = logging.getLogger("ercot_bench.tasks")


def _task_id(template_id: str, params: dict) -> str:
    h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return f"{template_id}-{h}"


def generate_for_template(tpl: Template, ctx: Ctx, n: int, seed: int, split: str, max_attempts_factor: int = 25) -> list[Task]:
    rng = random.Random(f"{seed}-{tpl.id}-{split}")
    tasks: dict[str, Task] = {}
    attempts = 0
    while len(tasks) < n and attempts < n * max_attempts_factor:
        attempts += 1
        params = tpl.sample_params(rng, ctx)
        if params is None:
            break  # nothing sampleable in this context
        tid = _task_id(tpl.id, params)
        if tid in tasks:
            continue
        answer = tpl.reference_answer(params, ctx)
        if answer is None:
            continue
        tasks[tid] = Task(
            task_id=tid, template_id=tpl.id, family=tpl.family, difficulty=tpl.difficulty, split=split,
            question=tpl.render_question(params), params=params, answer=answer, answer_type=tpl.answer_type,
            tolerance=Tolerance(**tpl.tolerance), answer_timezone=tpl.answer_timezone, ordered=tpl.ordered,
        )
    return list(tasks.values())


def generate_all(cfg: Config, n_per_template: int, templates: list[Template] | None = None) -> dict[str, list[Task]]:
    templates = templates or ALL_TEMPLATES
    con = connect_readonly(cfg.db_path)
    all_years = sorted({r[0] for r in con.execute("SELECT DISTINCT year(delivery_date) FROM da_spp_hourly").fetchall()})
    years = split_years(cfg.train_years, all_years)
    heldout = heldout_template_ids([t.id for t in ALL_TEMPLATES], cfg.seed, cfg.heldout_template_fraction)
    ctxs = {TRAIN: Ctx(con, years[TRAIN]), "test_in_template": Ctx(con, years["test_in_template"]),
            TEST_HELDOUT: Ctx(con, None)}
    out: dict[str, list[Task]] = {s: [] for s in SPLITS}
    for tpl in templates:
        if not tpl.available(ctxs[TEST_HELDOUT]):
            log.warning("template %s skipped: missing tables %s", tpl.id, tpl.requires)
            continue
        target_splits = [TEST_HELDOUT] if tpl.id in heldout else [TRAIN, "test_in_template"]
        for split in target_splits:
            tasks = generate_for_template(tpl, ctxs[split], n_per_template, cfg.seed, split)
            if len(tasks) < n_per_template:
                log.warning("template %s / %s: only %d of %d tasks", tpl.id, split, len(tasks), n_per_template)
            out[split].extend(tasks)
    con.close()
    return out


def write_splits(tasks: dict[str, list[Task]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, ts in tasks.items():
        p = out_dir / f"{split}.jsonl"
        with p.open("w") as f:
            for t in sorted(ts, key=lambda t: t.task_id):
                f.write(t.model_dump_json() + "\n")
        paths[split] = p
    return paths


def load_tasks(path: Path | str) -> list[Task]:
    with open(path) as f:
        return [Task.model_validate_json(line) for line in f if line.strip()]


def summarize(tasks: dict[str, list[Task]]) -> str:
    lines = []
    for split, ts in tasks.items():
        by_t = Counter(t.template_id for t in ts)
        by_d = Counter(t.difficulty for t in ts)
        lines.append(f"{split}: {len(ts)} tasks, {len(by_t)} templates, difficulty {dict(sorted(by_d.items()))}")
    return "\n".join(lines)
