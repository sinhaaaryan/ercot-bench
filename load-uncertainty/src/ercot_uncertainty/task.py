"""Framework-agnostic task logic: output parsing and reward.

The Verifiers env, TRL fallback, and eval harness all import from here.
Prompt building lives here too once the feature pipeline exists.

An example is one (weather zone, operating day). The model predicts p10/p50/p90 of
ERCOT day-ahead load-forecast error (actual - forecast, MW) for each peak hour-ending.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from ercot_uncertainty.baselines.scoring import mean_quantile_loss

# Reward constants
INVALID_REWARD = -1.0  # unparseable / wrong schema: as bad as the worst clipped score
UNORDERED_PENALTY = 0.1  # percentiles out of order are sorted, then penalised
MIN_BASELINE_LOSS_MW = 5.0  # floor so near-perfect baseline days don't blow up the ratio
MAX_ABS_MW = 50_000.0  # sanity bound on any predicted value

# Multiplier variant: model scales the baseline's interval around the baseline median
SCALE_MIN, SCALE_MAX = 0.25, 4.0

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
# A leading "+" on a number ("p50": +35) is invalid JSON but unambiguous; the prompt shows signed numbers, so
# models copy the style. Only matches a "+" directly after ':' '[' or ',' (value position), not inside strings
# like "+5 MW" preceded by other text.
_PLUS_RE = re.compile(r"([:\[,]\s*)\+(?=\d|\.\d)")


@dataclass
class Parsed:
    ok: bool
    hours: dict[int, dict[str, float]] = field(default_factory=dict)
    reordered: bool = False
    error: str | None = None


def _extract_json(text: str) -> Any:
    """Parse JSON from a completion. Tolerates code fences, surrounding prose and leading '+' on numbers.
    Structural errors (unbalanced braces, missing commas) still fail."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    text = _PLUS_RE.sub(r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def _as_number(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"not a number: {v!r}")
    x = float(v)
    if not math.isfinite(x) or abs(x) > MAX_ABS_MW:
        raise ValueError(f"out of range: {v!r}")
    return x


def parse_percentiles(text: str, expected_hours: list[int]) -> Parsed:
    """Parse {"hours":[{"he":17,"p10":..,"p50":..,"p90":..}, ...]}.

    Every expected hour must be present exactly once; extra hours are ignored.
    Out-of-order percentiles are sorted and flagged (penalised in the reward).
    """
    try:
        obj = _extract_json(text)
        rows = obj["hours"]
        if not isinstance(rows, list):
            raise ValueError("'hours' is not a list")
        hours: dict[int, dict[str, float]] = {}
        reordered = False
        for row in rows:
            he = int(_as_number(row["he"]))
            if he in hours:
                raise ValueError(f"duplicate he={he}")
            vals = [_as_number(row[k]) for k in ("p10", "p50", "p90")]
            if vals != sorted(vals):
                reordered = True
                vals.sort()
            hours[he] = dict(zip(("p10", "p50", "p90"), vals))
        missing = [h for h in expected_hours if h not in hours]
        if missing:
            raise ValueError(f"missing hours {missing}")
        return Parsed(True, {h: hours[h] for h in expected_hours}, reordered)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
        return Parsed(False, error=f"{type(e).__name__}: {e}")


def parse_scale(text: str) -> float | None:
    """Multiplier variant: {"scale": 1.3}. Returns clipped scale or None if invalid."""
    try:
        s = _as_number(_extract_json(text)["scale"])
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if s <= 0:
        return None
    return min(max(s, SCALE_MIN), SCALE_MAX)


def apply_scale(baseline: dict[int, dict[str, float]], scale: float) -> dict[int, dict[str, float]]:
    """Widen/tighten the baseline interval around its median."""
    out = {}
    for he, q in baseline.items():
        m = q["p50"]
        out[he] = {"p10": m + scale * (q["p10"] - m), "p50": m, "p90": m + scale * (q["p90"] - m)}
    return out


def relative_skill(
    pred: dict[int, dict[str, float]],
    baseline: dict[int, dict[str, float]],
    actual: dict[int, float],
) -> float:
    """clip(1 - model_loss / baseline_loss, -1, 1). 0 = baseline, 1 = perfect.

    Written as (base - model) / max(base, floor) so that on near-perfect baseline days the
    floor only shrinks the magnitude; being worse than the baseline is always negative.
    """
    hours = sorted(actual)
    ys = [actual[h] for h in hours]
    model_loss = mean_quantile_loss([pred[h] for h in hours], ys)
    base_loss = mean_quantile_loss([baseline[h] for h in hours], ys)
    skill = (base_loss - model_loss) / max(base_loss, MIN_BASELINE_LOSS_MW)
    return float(min(max(skill, -1.0), 1.0))


def scaled_skill(
    pred: dict[int, dict[str, float]],
    baseline: dict[int, dict[str, float]],
    actual: dict[int, float],
    scale: float,
) -> float:
    """clip((baseline_loss - model_loss) / scale, -1, 1): MW of pinball loss saved vs the baseline, in units of the
    zone's typical baseline loss. Unlike the per-example ratio, a volatile day (where most MW can be saved) weighs more
    than a calm one, and zones are balanced -- the same quantities the eval's headline metrics measure."""
    hours = sorted(actual)
    ys = [actual[h] for h in hours]
    model_loss = mean_quantile_loss([pred[h] for h in hours], ys)
    base_loss = mean_quantile_loss([baseline[h] for h in hours], ys)
    return float(min(max((base_loss - model_loss) / scale, -1.0), 1.0))


def reward_percentiles(
    completion: str,
    baseline: dict[int, dict[str, float]],
    actual: dict[int, float],
    scale: float | None = None,
) -> float:
    """RL reward. With `scale` (the example's zone-typical baseline loss, MW): scaled_skill; without: the per-example
    ratio relative_skill (original definition, kept for backwards compatibility)."""
    parsed = parse_percentiles(completion, sorted(actual))
    if not parsed.ok:
        return INVALID_REWARD
    r = scaled_skill(parsed.hours, baseline, actual, scale) if scale else relative_skill(parsed.hours, baseline, actual)
    if parsed.reordered:
        r -= UNORDERED_PENALTY
    return max(r, INVALID_REWARD)


def reward_scale(
    completion: str,
    baseline: dict[int, dict[str, float]],
    actual: dict[int, float],
) -> float:
    scale = parse_scale(completion)
    if scale is None:
        return INVALID_REWARD
    return relative_skill(apply_scale(baseline, scale), baseline, actual)
