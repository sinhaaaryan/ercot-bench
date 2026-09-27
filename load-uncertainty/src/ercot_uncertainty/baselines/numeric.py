"""Numeric baselines for the distribution of ERCOT day-ahead load-forecast error.

Input frame (one row per zone x operating_date x hour_ending):
    zone, operating_date, hour_ending, error (actual - dam forecast, MW), + feature columns.

1. historical: empirical quantiles of the same zone/hour's errors over a trailing window, using
   only operating days <= D - LAG_DAYS (D-1's actuals are not complete at the D-1 10:00 cutoff).
2. gbm: LightGBM quantile regression on numeric features (model spread, recent errors, weather...).
   Train-period predictions are out-of-fold over contiguous time blocks, so the baseline that
   normalises the RL reward is never scored in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ercot_uncertainty.baselines.scoring import QUANTILES

LAG_DAYS = 2
KEY = ["zone", "operating_date", "hour_ending"]


def historical_quantiles(df: pd.DataFrame, window_days: int = 90, min_obs: int = 20) -> pd.DataFrame:
    """Trailing-window empirical p10/p50/p90 per (zone, hour_ending), strictly causal."""
    df = df[KEY + ["error"]].copy()
    was_datetime = pd.api.types.is_datetime64_any_dtype(df["operating_date"])
    df["operating_date"] = pd.to_datetime(df["operating_date"])
    out = []
    for (zone, he), g in df.groupby(["zone", "hour_ending"], sort=False):
        g = g.sort_values("operating_date")
        s = g.set_index("operating_date")["error"]
        roll = s.rolling(f"{window_days}D", min_periods=min_obs)
        stats = pd.DataFrame({k: roll.quantile(tau) for k, tau in QUANTILES.items()}).reset_index()
        stats = stats.rename(columns={"operating_date": "stat_date"})
        # value for day D = trailing stats as of the latest observed day <= D - LAG_DAYS
        target = g[["operating_date"]].assign(lookup=lambda t: t["operating_date"] - pd.Timedelta(days=LAG_DAYS))
        m = pd.merge_asof(target, stats, left_on="lookup", right_on="stat_date", direction="backward")
        out.append(m.assign(zone=zone, hour_ending=he))
    res = pd.concat(out, ignore_index=True)
    if not was_datetime:
        res["operating_date"] = res["operating_date"].dt.date
    return res[KEY + list(QUANTILES)]


@dataclass
class QuantileGBM:
    features: list[str]
    categorical: list[str] = field(default_factory=lambda: ["zone"])
    params: dict = field(
        default_factory=lambda: dict(
            n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=80,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1,
        )
    )
    models: dict[str, lgb.LGBMRegressor] = field(default_factory=dict)

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.features].copy()
        for c in self.categorical:
            if c in x:
                x[c] = x[c].astype("category")
        return x

    def fit(self, df: pd.DataFrame) -> "QuantileGBM":
        x, y = self._x(df), df["error"].to_numpy()
        self.models = {
            k: lgb.LGBMRegressor(objective="quantile", alpha=tau, **self.params).fit(x, y)
            for k, tau in QUANTILES.items()
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._x(df)
        preds = np.column_stack([self.models[k].predict(x) for k in QUANTILES])
        preds.sort(axis=1)  # enforce p10 <= p50 <= p90 (quantile crossing)
        out = df[KEY].copy()
        out[list(QUANTILES)] = preds
        return out


def time_blocks(dates: pd.Series, n_blocks: int) -> np.ndarray:
    """Assign each row to one of n contiguous blocks of operating dates."""
    uniq = np.sort(pd.to_datetime(dates).unique())
    edges = np.array_split(uniq, n_blocks)
    lookup = {d: i for i, blk in enumerate(edges) for d in blk}
    return pd.to_datetime(dates).map(lookup).to_numpy()


# Upgraded config from the baseline audit (scripts/audit_baseline*.py): bigger trees + per-zone conformal calibration.
UPGRADED_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


def _conformal_adjust(pred: pd.DataFrame, truth: pd.DataFrame, level: float = 0.8) -> pd.Series:
    """Per-zone widening a (can be negative = narrowing) so that [p10 - a, p90 + a] covers `level` of `truth`
    (conformalized quantile regression). `pred` and `truth` must be aligned out-of-sample predictions/labels."""
    m = pred.merge(truth[KEY + ["error"]], on=KEY)
    nonconf = np.maximum(m["p10"] - m["error"], m["error"] - m["p90"])
    return nonconf.groupby(m["zone"]).quantile(level)


def _apply(pred: pd.DataFrame, adj: pd.Series) -> pd.DataFrame:
    out = pred.copy()
    a = out["zone"].map(adj).fillna(0.0).to_numpy()
    out["p10"] -= a
    out["p90"] += a
    out[list(QUANTILES)] = np.sort(out[list(QUANTILES)].to_numpy(), axis=1)
    return out


def gbm_oof_and_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    n_blocks: int = 5,
    embargo_days: int = 7,
    params: dict | None = None,
    categorical: list[str] | None = None,
    calibrate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold predictions for `train` (blocked CV with an embargo) + fit-on-all for `test`.

    calibrate=True adds per-zone conformal calibration of the p10-p90 interval to 80% coverage, itself out-of-fold:
    block b's adjustment is estimated from the other blocks' out-of-fold predictions; test uses all train blocks.
    """
    def model():
        m = QuantileGBM(features, categorical=categorical or ["zone"])
        if params:
            m.params = dict(params)
        return m

    blocks = time_blocks(train["operating_date"], n_blocks)
    dates = pd.to_datetime(train["operating_date"])
    oof = []
    for b in range(n_blocks):
        held = blocks == b
        lo, hi = dates[held].min(), dates[held].max()
        emb = pd.Timedelta(days=embargo_days)
        fit_mask = ~held & ((dates < lo - emb) | (dates > hi + emb))
        oof.append(model().fit(train[fit_mask]).predict(train[held]).assign(_block=b))
    oof = pd.concat(oof, ignore_index=True)
    test_pred = model().fit(train).predict(test)
    if calibrate:
        truth = train[KEY + ["error"]]
        parts = []
        for b in range(n_blocks):
            adj_b = _conformal_adjust(oof[oof["_block"] != b], truth)
            parts.append(_apply(oof[oof["_block"] == b], adj_b))
        test_pred = _apply(test_pred, _conformal_adjust(oof, truth))
        oof = pd.concat(parts, ignore_index=True)
    return oof.drop(columns="_block"), test_pred
