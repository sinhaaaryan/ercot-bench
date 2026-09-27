"""The verifiers reward must equal our harness reward on the same completions.

Replays every recorded completion in ../../results/*.jsonl (written by `ercot-bench eval`) through
ErcotSqlTask's @vf.reward method and compares with the reward the harness logged.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ercot_bench.tasks.generator import load_tasks
from ercot_sql.taskset import ErcotSqlConfig, ErcotSqlTaskset

ROOT = Path(__file__).resolve().parents[3]
RESULTS = sorted((ROOT / "results").glob("*.jsonl"))


@pytest.mark.skipif(not RESULTS, reason="no eval results to replay")
def test_verifiers_reward_equals_harness_reward():
    rows = [json.loads(l) for f in RESULTS for l in open(f)]
    tasks = {}
    for split in {r["split"] for r in rows}:
        ts = ErcotSqlTaskset(ErcotSqlConfig(id="ercot-sql", split=split))
        tasks.update({t.data.task_id: t for t in ts.load()})

    async def run():
        out = []
        for r in rows:
            t = tasks[r["task_id"]]
            trace = SimpleNamespace(last_reply=r["completion"].strip())
            out.append((await t.ercot_sql_reward(trace), await t.outcome(trace)))
        return out

    got = asyncio.run(run())
    mismatches = [(r["task_id"], r["reward"], g) for r, (g, _) in zip(rows, got) if abs(r["reward"] - g) > 1e-9]
    assert not mismatches, mismatches[:5]
    assert len(rows) > 0


def test_taskset_loads_with_same_prompt():
    from ercot_bench.config import load_config
    from ercot_bench.env.prompt import build_prompt

    cfg = load_config()
    ts = ErcotSqlTaskset(ErcotSqlConfig(id="ercot-sql", split="test_in_template", subset_size=5))
    loaded = ts.load()
    assert len(loaded) == 5
    raw = {t.task_id: t for t in load_tasks(cfg.tasks_dir / "test_in_template.jsonl")}
    for t in loaded:
        system, user = build_prompt(raw[t.data.task_id].question, cfg.db_path)
        assert t.data.system_prompt == system and t.data.prompt == user
    assert len({t.key for t in loaded}) == 5
