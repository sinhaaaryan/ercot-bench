"""The verifiers taskset must score exactly like ercot_uncertainty.task.reward_percentiles.

Runs on CPU, no model: builds tasks from the synthetic smoke split, feeds hand-written completions
through the real Task.score() path (the one the env server uses) and compares.

    PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env \
        /hackathon/prime-rl/.venv/bin/python -m pytest -q environments/ercot_uncertainty_env/tests
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers.v1")

from ercot_uncertainty.baselines.scoring import coverage, mean_quantile_loss  # noqa: E402
from ercot_uncertainty.task import INVALID_REWARD, UNORDERED_PENALTY, reward_percentiles  # noqa: E402
from ercot_uncertainty_env.taskset import (  # noqa: E402
    ErcotUncertaintyConfig,
    ErcotUncertaintyTaskset,
    example_to_data,
    load_examples,
    split_path,
)

ROOT = Path(__file__).resolve().parents[3]
SMOKE = ROOT / "data" / "examples" / "smoke"


def _taskset(split: str, **kw) -> ErcotUncertaintyTaskset:
    return ErcotUncertaintyTaskset(ErcotUncertaintyConfig(id="ercot-uncertainty-env", split=split, **kw))


def _score(task, text: str):
    trace = vf.Trace.model_construct(root_reply=text, rewards={}, metrics={})
    asyncio.run(task.score(trace))
    return trace


def _as_json(rows: dict[int, tuple[float, float, float]]) -> str:
    return json.dumps({"hours": [{"he": h, "p10": a, "p50": b, "p90": c} for h, (a, b, c) in rows.items()]})


def _completions(row: dict) -> dict[str, str]:
    hours = [int(h) for h in row["hours"]]
    base = {int(k): v for k, v in row["baseline"].items()}
    act = {int(k): v for k, v in row["actual_error"].items()}
    return {
        "baseline": _as_json({h: (base[h]["p10"], base[h]["p50"], base[h]["p90"]) for h in hours}),
        "perfect": _as_json({h: (act[h], act[h], act[h]) for h in hours}),
        "tight": _as_json({h: (-100, 0, 100) for h in hours}),
        "wide": _as_json({h: (-5000, 0, 5000) for h in hours}),
        "swapped": _as_json({h: (base[h]["p90"], base[h]["p50"], base[h]["p10"]) for h in hours}),
        "fenced": "Here you go:\n```json\n" + _as_json({h: (-300, 50, 400) for h in hours}) + "\n```",
        "missing_hour": _as_json({h: (-1, 0, 1) for h in hours[:-1]}),
        "garbage": "I think the load will be high tomorrow.",
        "empty": "",
    }


@pytest.fixture(scope="module")
def smoke_rows():
    path = split_path("smoke/val", SMOKE.parent)
    if not path.exists():
        pytest.skip("run scripts/make_smoke_examples.py first")
    return load_examples(path)


def test_loads_via_verifiers_plugin_loader():
    from verifiers.v1.utils.loaders import taskset_class

    assert taskset_class("ercot-uncertainty-env") is ErcotUncertaintyTaskset


def test_prompt_passthrough(smoke_rows):
    tasks = list(_taskset("smoke/val"))
    assert len(tasks) == len(smoke_rows)
    for t, r in zip(tasks, smoke_rows):
        assert t.key == r["id"]
        assert t.data.system_prompt == r["prompt"][0]["content"]
        assert t.data.prompt == r["prompt"][1]["content"]
    assert len({t.key for t in tasks}) == len(tasks)


def test_rewards_match_task_py(smoke_rows):
    tasks = {t.key: t for t in _taskset("smoke/val")}
    n = 0
    for row in smoke_rows:
        baseline = {int(k): v for k, v in row["baseline"].items()}
        actual = {int(k): float(v) for k, v in row["actual_error"].items()}
        for name, text in _completions(row).items():
            trace = _score(tasks[row["id"]], text)
            expected = reward_percentiles(text, baseline, actual)
            assert trace.rewards["pinball_skill"].value == pytest.approx(expected, abs=1e-12), (row["id"], name)
            assert trace.reward == pytest.approx(expected, abs=1e-12)
            n += 1
    assert n == len(smoke_rows) * 9


def test_reward_anchor_points(smoke_rows):
    task = next(iter(_taskset("smoke/val", limit=1)))
    c = _completions(smoke_rows[0])
    assert _score(task, c["baseline"]).reward == pytest.approx(0.0, abs=1e-9)
    assert _score(task, c["perfect"]).reward == pytest.approx(1.0)
    assert _score(task, c["swapped"]).reward == pytest.approx(-UNORDERED_PENALTY)
    for bad in ("missing_hour", "garbage", "empty"):
        assert _score(task, c[bad]).reward == INVALID_REWARD


def test_metrics(smoke_rows):
    row = smoke_rows[0]
    task = next(iter(_taskset("smoke/val", limit=1)))
    hours = [int(h) for h in row["hours"]]
    ys = [float(row["actual_error"][str(h)]) for h in hours]
    c = _completions(row)

    m = _score(task, c["wide"]).metrics
    assert m["parse_ok"] == 1.0 and m["reordered"] == 0.0
    assert m["coverage"] == pytest.approx(coverage([{"p10": -5000, "p50": 0, "p90": 5000}] * len(hours), ys))
    assert m["pinball_mw"] == pytest.approx(mean_quantile_loss([{"p10": -5000, "p50": 0, "p90": 5000}] * len(hours), ys))
    assert m["width_mw"] == pytest.approx(10_000)
    base = [row["baseline"][str(h)] for h in hours]
    assert m["baseline_pinball_mw"] == pytest.approx(mean_quantile_loss(base, ys))

    m = _score(task, c["perfect"]).metrics
    assert m["pinball_mw"] == pytest.approx(0.0) and m["coverage"] == 1.0 and m["width_mw"] == 0.0

    assert _score(task, c["swapped"]).metrics["reordered"] == 1.0

    m = _score(task, c["garbage"]).metrics
    assert m["parse_ok"] == 0.0 and m["reordered"] == 0.0
    assert "coverage" not in m and "pinball_mw" not in m  # conditional-on-parse metrics are omitted


def test_string_hour_keys_and_extra_label_hours_are_handled():
    row = {
        "id": "x", "zone": "north", "operating_date": "2024-07-01", "split": "t", "hours": [17, 16],
        "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "baseline": {"16": {"p10": -1, "p50": 0, "p90": 1}, "17": {"p10": -2, "p50": 0, "p90": 2},
                     "18": {"p10": -3, "p50": 0, "p90": 3}},
        "actual_error": {"16": 5.0, "17": -5.0, "18": 100.0},
    }
    d = example_to_data(row, 0)
    assert d.hours == [17, 16] and sorted(d.actual_error) == ["16", "17"]  # only listed hours are scored
    with pytest.raises(ValueError):
        example_to_data({**row, "hours": [16, 19]}, 0)


def test_filters():
    assert len(list(_taskset("smoke/train", limit=5))) == 5
    only = list(_taskset("smoke/train", zones=["coast", "west"]))
    assert only and {t.data.zone for t in only} == {"coast", "west"}
