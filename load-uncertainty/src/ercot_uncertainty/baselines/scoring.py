"""Proper scoring for quantile forecasts of load-forecast error (MW).

Everything here is framework-agnostic and shared by baselines, eval, and the RL reward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

QUANTILES: dict[str, float] = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


def pinball(y: float | np.ndarray, q_pred: float | np.ndarray, tau: float) -> np.ndarray:
    """Pinball (quantile) loss: tau*(y-q) if y>=q else (1-tau)*(q-y)."""
    diff = np.asarray(y, dtype=float) - np.asarray(q_pred, dtype=float)
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def quantile_loss(pred: Mapping[str, float], y: float) -> float:
    """Mean pinball loss over p10/p50/p90 for a single target value."""
    return float(np.mean([pinball(y, pred[k], tau) for k, tau in QUANTILES.items()]))


def mean_quantile_loss(preds: Sequence[Mapping[str, float]], ys: Sequence[float]) -> float:
    """Mean pinball loss over hours (or any list of targets) and quantiles."""
    if len(preds) != len(ys) or not preds:
        raise ValueError(f"need equal, non-empty preds/ys (got {len(preds)}, {len(ys)})")
    return float(np.mean([quantile_loss(p, y) for p, y in zip(preds, ys)]))


def coverage(preds: Sequence[Mapping[str, float]], ys: Sequence[float]) -> float:
    """Fraction of targets inside [p10, p90] (nominal 80%)."""
    inside = [p["p10"] <= y <= p["p90"] for p, y in zip(preds, ys)]
    return float(np.mean(inside))


def interval_width(preds: Sequence[Mapping[str, float]]) -> float:
    return float(np.mean([p["p90"] - p["p10"] for p in preds]))
