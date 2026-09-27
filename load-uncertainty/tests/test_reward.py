import json

import numpy as np
import pytest

from ercot_uncertainty.baselines.scoring import coverage, mean_quantile_loss, pinball
from ercot_uncertainty.task import (
    INVALID_REWARD,
    SCALE_MAX,
    UNORDERED_PENALTY,
    apply_scale,
    parse_percentiles,
    parse_scale,
    relative_skill,
    reward_percentiles,
    reward_scale,
)

HOURS = [16, 17, 18]


def q(p10, p50, p90):
    return {"p10": p10, "p50": p50, "p90": p90}


def as_json(pred: dict[int, dict[str, float]]) -> str:
    return json.dumps({"hours": [{"he": h, **v} for h, v in pred.items()]})


BASELINE = {h: q(-800, 0, 800) for h in HOURS}


# --- pinball -------------------------------------------------------------------------------


def test_pinball_asymmetry():
    assert pinball(100, 0, 0.9) == pytest.approx(90)  # under-predicted high quantile: costly
    assert pinball(-100, 0, 0.9) == pytest.approx(10)
    assert pinball(5, 5, 0.1) == 0


def test_pinball_minimised_at_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 500, 20_000)
    grid = np.linspace(-1500, 1500, 301)
    for tau in (0.1, 0.5, 0.9):
        losses = [pinball(y, c, tau).mean() for c in grid]
        best = grid[int(np.argmin(losses))]
        assert best == pytest.approx(np.quantile(y, tau), abs=25)


# --- reward properties ---------------------------------------------------------------------


def test_baseline_scores_zero():
    actual = {h: 300.0 for h in HOURS}
    assert reward_percentiles(as_json(BASELINE), BASELINE, actual) == pytest.approx(0)


def test_perfect_prediction_scores_best():
    actual = {16: 120.0, 17: -40.0, 18: 900.0}
    perfect = {h: q(y, y, y) for h, y in actual.items()}
    r_perfect = reward_percentiles(as_json(perfect), BASELINE, actual)
    assert r_perfect == pytest.approx(1.0)
    for other in (BASELINE, {h: q(y - 50, y, y + 50) for h, y in actual.items()}):
        assert reward_percentiles(as_json(other), BASELINE, actual) < r_perfect


def test_wide_intervals_penalised_on_calm_day():
    actual = {h: 20.0 for h in HOURS}  # ERCOT nailed it
    tight = {h: q(-200, 0, 200) for h in HOURS}
    wide = {h: q(-3000, 0, 3000) for h in HOURS}
    assert reward_percentiles(as_json(tight), BASELINE, actual) > 0
    assert reward_percentiles(as_json(wide), BASELINE, actual) < 0


def test_always_wide_is_not_a_winning_strategy():
    """Over a realistic mix of days, day-specific widths beat a constant wide hedge."""
    rng = np.random.default_rng(1)
    scores_adaptive, scores_wide = [], []
    for i in range(400):
        sigma = 150 if i % 4 else 1500  # 3 calm days per volatile day
        actual = {h: float(rng.normal(0, sigma)) for h in HOURS}
        z = 1.2816  # N(0,1) 90th percentile
        adaptive = {h: q(-z * sigma, 0, z * sigma) for h in HOURS}
        wide = {h: q(-z * 1500, 0, z * 1500) for h in HOURS}
        scores_adaptive.append(reward_percentiles(as_json(adaptive), BASELINE, actual))
        scores_wide.append(reward_percentiles(as_json(wide), BASELINE, actual))
    assert np.mean(scores_adaptive) > np.mean(scores_wide) + 0.1


def test_reward_clipped():
    actual = {h: 0.0 for h in HOURS}
    awful = {h: q(40_000, 45_000, 49_000) for h in HOURS}
    assert reward_percentiles(as_json(awful), BASELINE, actual) == -1.0


def test_tiny_baseline_loss_is_floored():
    base = {h: q(0, 0, 0) for h in HOURS}
    actual = {h: 0.0 for h in HOURS}
    slightly_off = {h: q(-1, 0, 1) for h in HOURS}
    r = relative_skill(slightly_off, base, actual)
    assert -1.0 <= r < 0  # finite, no division blow-up


# --- parsing edge cases --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json",
        "{}",
        '{"hours": {}}',
        '{"hours": [{"he": 16, "p10": 1, "p50": 2}]}',  # missing p90 and hours
        '{"hours": [{"he": 16, "p10": "a", "p50": 2, "p90": 3}]}',
        '{"hours": [{"he": 16, "p10": NaN, "p50": 2, "p90": 3}]}',
        '{"hours": [{"he": 16, "p10": 1, "p50": 2, "p90": 1e12}]}',
        '{"hours": [{"he": 16, "p10": true, "p50": 2, "p90": 3}]}',
        "[1, 2, 3]",
    ],
)
def test_invalid_outputs_get_fixed_penalty(text):
    actual = {h: 0.0 for h in HOURS}
    assert reward_percentiles(text, BASELINE, actual) == INVALID_REWARD


def test_missing_and_duplicate_hours_invalid():
    rows = [{"he": 16, **q(-1, 0, 1)}, {"he": 17, **q(-1, 0, 1)}]
    assert not parse_percentiles(json.dumps({"hours": rows}), HOURS).ok
    rows.append({"he": 17, **q(-1, 0, 1)})
    rows.append({"he": 18, **q(-1, 0, 1)})
    assert not parse_percentiles(json.dumps({"hours": rows}), HOURS).ok


def test_code_fence_prose_and_extra_hours_accepted():
    body = as_json({**BASELINE, 19: q(0, 0, 0)})
    for text in (f"```json\n{body}\n```", f"Here you go:\n{body}\nThanks", body):
        p = parse_percentiles(text, HOURS)
        assert p.ok and sorted(p.hours) == HOURS


def test_numeric_strings_and_int_he_accepted():
    text = '{"hours":[' + ",".join(f'{{"he":"{h}","p10":"-5","p50":0,"p90":5.5}}' for h in HOURS) + "]}"
    assert parse_percentiles(text, HOURS).ok


def test_unordered_percentiles_sorted_and_penalised():
    actual = {h: 0.0 for h in HOURS}
    swapped = json.dumps({"hours": [{"he": h, "p10": 800, "p50": 0, "p90": -800} for h in HOURS]})
    p = parse_percentiles(swapped, HOURS)
    assert p.ok and p.reordered and p.hours[16] == q(-800, 0, 800)
    assert reward_percentiles(swapped, BASELINE, actual) == pytest.approx(-UNORDERED_PENALTY)


# --- multiplier variant --------------------------------------------------------------------


def test_scale_parse_and_clip():
    assert parse_scale('{"scale": 1.5}') == 1.5
    assert parse_scale('{"scale": 99}') == SCALE_MAX
    assert parse_scale('{"scale": 0}') is None
    assert parse_scale('{"scale": -1}') is None
    assert parse_scale("nope") is None


def test_scale_one_equals_baseline():
    actual = {h: 1000.0 for h in HOURS}
    assert apply_scale(BASELINE, 1.0) == BASELINE
    assert reward_scale('{"scale": 1.0}', BASELINE, actual) == pytest.approx(0)


def test_scale_rewards_right_direction():
    calm = {h: 10.0 for h in HOURS}
    wild = {h: 2500.0 for h in HOURS}
    assert reward_scale('{"scale": 0.5}', BASELINE, calm) > 0
    assert reward_scale('{"scale": 0.5}', BASELINE, wild) < 0
    assert reward_scale('{"scale": 2.5}', BASELINE, wild) > 0
    assert reward_scale("garbage", BASELINE, calm) == INVALID_REWARD


def test_coverage_and_loss_helpers():
    preds = [q(-1, 0, 1), q(-1, 0, 1)]
    assert coverage(preds, [0, 5]) == 0.5
    assert mean_quantile_loss(preds, [0, 0]) == pytest.approx((0.1 + 0 + 0.1) / 3)


def test_leading_plus_signs_accepted():
    body = '{"hours":[' + ",".join(f'{{"he":{h},"p10":-120,"p50":+35,"p90": +180.5}}' for h in HOURS) + "]}"
    p = parse_percentiles(f"```json\n{body}\n```", HOURS)
    assert p.ok and p.hours[16] == q(-120, 35, 180.5)
    pretty = '{\n "hours": [\n' + ",\n".join(f'  {{"he": {h}, "p10": -1, "p50": +2, "p90": +3}}' for h in HOURS) + "\n ]\n}"
    assert parse_percentiles(pretty, HOURS).ok


def test_structural_errors_still_fail():
    rows = ",".join(f'{{"he":{h},"p10":-1,"p50":0,"p90":1}}' for h in HOURS)
    assert not parse_percentiles('{"hours":[' + rows + "]}}", HOURS).ok  # extra brace
    assert not parse_percentiles('```json\n{"hours":[' + rows + "]\n```", HOURS).ok  # missing brace


# --- zone-scaled reward (RL): MW saved / zone-typical baseline loss ------------------------------------------------------


def test_scaled_reward_echo_is_zero_and_sign_is_right():
    actual = {16: 300.0, 17: -200.0, 18: 50.0}
    assert reward_percentiles(as_json(BASELINE), BASELINE, actual, scale=100.0) == pytest.approx(0)
    better = {h: q(y - 100, y, y + 100) for h, y in actual.items()}
    worse = {h: q(-3000, 0, 3000) for h in actual}
    assert reward_percentiles(as_json(better), BASELINE, actual, scale=100.0) > 0
    assert reward_percentiles(as_json(worse), BASELINE, actual, scale=100.0) < 0


def test_scaled_reward_is_proportional_to_mw_saved():
    """Same % improvement: a volatile day (large baseline loss) earns more than a calm one; same MW -> same reward."""
    from ercot_uncertainty.task import scaled_skill
    calm_actual, wild_actual = {h: 0.0 for h in HOURS}, {h: 3000.0 for h in HOURS}
    calm_base = {h: q(-100, 0, 100) for h in HOURS}
    wild_base = {h: q(-100, 0, 100) for h in HOURS}
    calm_pred = {h: q(-90, 0, 90) for h in HOURS}           # 10% tighter on a calm day
    wild_pred = {h: q(-100, 300, 100 + 300) for h in HOURS}  # shift toward a big miss
    r_calm = scaled_skill(calm_pred, calm_base, calm_actual, 100.0)
    r_wild = scaled_skill(wild_pred, wild_base, wild_actual, 100.0)
    assert 0 < r_calm < r_wild


def test_scaled_reward_clipped_and_invalid_penalty_unchanged():
    actual = {h: 5000.0 for h in HOURS}
    perfect = {h: q(5000, 5000, 5000) for h in HOURS}
    assert reward_percentiles(as_json(perfect), BASELINE, actual, scale=10.0) == 1.0
    assert reward_percentiles("garbage", BASELINE, actual, scale=10.0) == INVALID_REWARD
    swapped = json.dumps({"hours": [{"he": h, "p10": 800, "p50": 0, "p90": -800} for h in HOURS]})
    assert reward_percentiles(swapped, BASELINE, {h: 0.0 for h in HOURS}, scale=100.0) == pytest.approx(-UNORDERED_PENALTY)
