"""ERCOT forecast-uncertainty predictor as a verifiers v1 taskset (single turn).

Each task is one (weather zone, operating day) example from data/examples/<split>.jsonl
(format: docs/DATA_CONTRACT.md). The prompt is the example's own `prompt` messages, unchanged.
The reward is `ercot_uncertainty.task.reward_percentiles` -- the same function the eval harness
and tests use; nothing is reimplemented here. Run it with the tool-less `null` harness:

    vf-eval ercot-uncertainty-env --env.agent.harness.id null --env.agent.runtime.type subprocess \
        --env.taskset.split smoke/val ...

Taskset id `ercot-uncertainty-env` imports the module `ercot_uncertainty_env` (verifiers maps
`-` to `_`); the id must NOT be `ercot-uncertainty`, which would import the task package itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import verifiers.v1 as vf

from ercot_uncertainty.baselines.scoring import coverage, interval_width, mean_quantile_loss
from ercot_uncertainty.task import parse_percentiles, reward_percentiles

# <repo>/environments/ercot_uncertainty_env/ercot_uncertainty_env/taskset.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = Path(os.environ.get("ERCOT_UNCERTAINTY_DATA_DIR", REPO_ROOT / "data" / "examples"))


class ErcotUncertaintyData(vf.TaskData):
    example_id: str
    zone: str
    operating_date: str
    split: str
    hours: list[int]
    baseline: dict[str, dict[str, float]]
    """Numeric-baseline percentiles (MW) keyed by hour-ending as a string (JSON keys)."""
    actual_error: dict[str, float]
    """Label: actual - ERCOT day-ahead forecast (MW), keyed by hour-ending as a string."""
    reward_scale: float | None = None
    """Zone-typical baseline pinball loss (MW); when set, the reward is MW saved / this scale (task.scaled_skill)."""


def _int_keys(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


class ErcotUncertaintyTask(vf.Task[ErcotUncertaintyData, vf.State, vf.TaskConfig]):
    @property
    def key(self) -> str:
        return self.data.example_id

    def _labels(self) -> tuple[dict[int, dict[str, float]], dict[int, float]]:
        return _int_keys(self.data.baseline), {h: float(v) for h, v in _int_keys(self.data.actual_error).items()}

    @vf.reward
    async def pinball_skill(self, trace: vf.Trace) -> float:
        """clip(1 - model_pinball / baseline_pinball, -1, 1); -1 for unparseable output;
        -0.1 extra if percentiles were out of order (ercot_uncertainty.task.reward_percentiles)."""
        baseline, actual = self._labels()
        return reward_percentiles(trace.last_reply, baseline, actual, self.data.reward_scale)

    @vf.metric
    async def forecast(self, trace: vf.Trace) -> dict[str, float]:
        """parse_ok / reordered on every rollout. coverage (share of hours with actual inside
        [p10, p90], nominal 0.8), pinball_mw (model mean pinball loss, MW), width_mw (mean p90-p10)
        only on parseable rollouts, so their means are conditional on parse_ok. baseline_pinball_mw
        is the example's baseline loss (same for every rollout of a task)."""
        baseline, actual = self._labels()
        hours = sorted(actual)
        ys = [actual[h] for h in hours]
        out = {"baseline_pinball_mw": mean_quantile_loss([baseline[h] for h in hours], ys)}
        parsed = parse_percentiles(trace.last_reply, hours)
        out["parse_ok"] = float(parsed.ok)
        out["reordered"] = float(parsed.ok and parsed.reordered)
        if parsed.ok:
            preds = [parsed.hours[h] for h in hours]
            out["coverage"] = coverage(preds, ys)
            out["pinball_mw"] = mean_quantile_loss(preds, ys)
            out["width_mw"] = interval_width(preds)
        return out


class ErcotUncertaintyConfig(vf.TasksetConfig):
    split: str = "train"
    """train | val | test | smoke/train | smoke/val (relative to data_dir, `.jsonl` implied), or a path to a JSONL."""
    data_dir: str = str(DEFAULT_DATA_DIR)
    """Directory holding <split>.jsonl (default: <repo>/data/examples or $ERCOT_UNCERTAINTY_DATA_DIR)."""
    limit: int | None = None
    """Keep only the first N examples (file order)."""
    zones: list[str] | None = None
    """Keep only these zone keys (e.g. ["north_central", "coast"])."""


def split_path(split: str, data_dir: str | Path) -> Path:
    if split.endswith(".jsonl"):
        return Path(split)
    return Path(data_dir) / f"{split}.jsonl"


def example_to_data(row: dict, idx: int) -> ErcotUncertaintyData:
    """One DATA_CONTRACT example -> task data. A leading system message becomes system_prompt;
    a single remaining user message becomes a plain-string prompt (what the `null` harness
    expects), otherwise the remaining messages are passed through as-is."""
    msgs = [dict(m) for m in row["prompt"]]
    system = None
    if msgs and msgs[0]["role"] == "system":
        system = msgs.pop(0)["content"]
    prompt = msgs[0]["content"] if len(msgs) == 1 and msgs[0]["role"] == "user" else msgs
    hours = [int(h) for h in row["hours"]]
    actual = {str(int(k)): float(v) for k, v in row["actual_error"].items()}
    baseline = {str(int(k)): {q: float(v[q]) for q in ("p10", "p50", "p90")} for k, v in row["baseline"].items()}
    missing = [h for h in hours if str(h) not in actual or str(h) not in baseline]
    if missing:
        raise ValueError(f"{row.get('id')}: hours {missing} lack baseline/actual_error")
    # score exactly the listed hours
    actual = {str(h): actual[str(h)] for h in hours}
    return ErcotUncertaintyData(
        idx=idx,
        name=row["id"],
        prompt=prompt,
        system_prompt=system,
        example_id=row["id"],
        zone=row["zone"],
        operating_date=row["operating_date"],
        split=row.get("split", ""),
        hours=hours,
        baseline=baseline,
        actual_error=actual,
        reward_scale=row.get("reward_scale"),
    )


def load_examples(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class ErcotUncertaintyTaskset(vf.Taskset[ErcotUncertaintyTask, ErcotUncertaintyConfig]):
    def load(self) -> list[ErcotUncertaintyTask]:
        c = self.config
        rows = load_examples(split_path(c.split, c.data_dir))
        if c.zones:
            rows = [r for r in rows if r["zone"] in set(c.zones)]
        if c.limit is not None:
            rows = rows[: c.limit]
        return [ErcotUncertaintyTask(example_to_data(r, i), c.task) for i, r in enumerate(rows)]
