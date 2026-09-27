"""ERCOT-Bench as a verifiers v1 taskset.

Single turn: the prompt is exactly ercot_bench.env.prompt.build_prompt(); run it with the tool-less
`null` harness. The reward calls ercot_bench.env.score.score() -- the same function our own eval
harness and SFT filter use. Scoring is not reimplemented here.

    uv run eval ercot-sql --env.agent.harness.id null --env.taskset.split test_in_template ...
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import verifiers.v1 as vf

from ercot_bench.config import load_config
from ercot_bench.env.prompt import build_prompt
from ercot_bench.env.score import score
from ercot_bench.eval.run import stratified_subset
from ercot_bench.tasks.generator import load_tasks
from ercot_bench.tasks.schema import Task as BenchTask

_CFG = load_config()


class ErcotSqlData(vf.TaskData):
    task_id: str
    template_id: str
    family: str
    difficulty: int
    split: str
    bench_task: str  # the full ercot_bench Task as JSON (answer, tolerance, answer type, ...)


class ErcotSqlTaskConfig(vf.TaskConfig):
    db_path: str = str(_CFG.db_path)
    """DuckDB file the model's SQL is executed against."""
    query_timeout_s: float = _CFG.query_timeout_s


class ErcotSqlTask(vf.Task[ErcotSqlData, vf.State, ErcotSqlTaskConfig]):
    @property
    def key(self) -> str:
        # unique even when a weighted RL task file repeats a task
        return f"{self.data.task_id}#{self.data.idx}"

    async def _score(self, trace: vf.Trace):
        task = BenchTask.model_validate_json(self.data.bench_task)
        return await asyncio.to_thread(score, trace.last_reply, task, self.config.db_path, self.config.query_timeout_s)

    @vf.reward
    async def ercot_sql_reward(self, trace: vf.Trace) -> float:
        """1.0 correct, 0.0 wrong/sql_error, -0.2 format_error (ercot_bench.env.score)."""
        return (await self._score(trace)).reward

    @vf.metric
    async def outcome(self, trace: vf.Trace) -> dict[str, float]:
        o = (await self._score(trace)).outcome
        return {f"is_{k}": float(o == k) for k in ("correct", "wrong_answer", "sql_error", "format_error")}


class ErcotSqlConfig(vf.TasksetConfig):
    split: str = "train"
    """train | test_in_template | test_heldout_templates, or a path to a tasks JSONL."""
    tasks_dir: str = str(_CFG.tasks_dir)
    db_path: str = str(_CFG.db_path)
    """DuckDB file used for the schema text in the prompt (and the default for task.db_path)."""
    subset_size: int | None = None
    """Stratified subset (round-robin over templates) of this many tasks."""
    task: ErcotSqlTaskConfig = ErcotSqlTaskConfig()


class ErcotSqlTaskset(vf.Taskset[ErcotSqlTask, ErcotSqlConfig]):
    def load(self) -> list[ErcotSqlTask]:
        c = self.config
        path = Path(c.split) if c.split.endswith(".jsonl") else Path(c.tasks_dir) / f"{c.split}.jsonl"
        tasks = stratified_subset(load_tasks(path), c.subset_size)
        out = []
        for i, t in enumerate(tasks):
            system, user = build_prompt(t.question, c.db_path)
            data = ErcotSqlData(idx=i, name=t.task_id, prompt=user, system_prompt=system, task_id=t.task_id,
                                template_id=t.template_id, family=t.family, difficulty=t.difficulty, split=t.split,
                                bench_task=t.model_dump_json())
            out.append(ErcotSqlTask(data, c.task))
        return out
