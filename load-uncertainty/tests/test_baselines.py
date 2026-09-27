import numpy as np
import pandas as pd
import pytest

from ercot_uncertainty.baselines.numeric import gbm_oof_and_test, historical_quantiles
from ercot_uncertainty.baselines.scoring import coverage


def synth(n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        spread = rng.gamma(2.0, 150.0)
        for zone in ("north_central", "coast"):
            for he in (17, 18):
                rows.append((zone, d, he, spread, rng.normal(0, 100 + 2.0 * spread)))
    return pd.DataFrame(rows, columns=["zone", "operating_date", "hour_ending", "spread", "error"])


def test_historical_is_causal():
    df = synth(200)
    q = historical_quantiles(df, window_days=30, min_obs=10)
    # change future errors drastically; past predictions must not move
    df2 = df.copy()
    cut = df2["operating_date"] >= df2["operating_date"].iloc[0] + pd.Timedelta(days=100)
    df2.loc[cut, "error"] += 1e5
    q2 = historical_quantiles(df2, window_days=30, min_obs=10)
    early = q["operating_date"] < (df["operating_date"].iloc[0] + pd.Timedelta(days=100 + 2))
    pd.testing.assert_frame_equal(q[early].reset_index(drop=True), q2[early].reset_index(drop=True))
    # and the first days have no history -> NaN
    assert q.sort_values("operating_date")["p50"].isna().iloc[0]


def test_historical_lag_excludes_d_minus_1():
    df = synth(60)
    d = sorted(df["operating_date"].unique())
    df2 = df.copy()
    df2.loc[df2["operating_date"] == d[40], "error"] += 1e6  # huge error on day 40
    idx = ["zone", "hour_ending", "operating_date"]
    q0 = historical_quantiles(df, window_days=30, min_obs=5).set_index(idx)
    q = historical_quantiles(df2, window_days=30, min_obs=5).set_index(idx)
    # day 41 (D-1 = day 40) must not see it; day 42 (D-2 = day 40) must
    k41 = ("coast", 17, d[41])
    k42 = ("coast", 17, d[42])
    assert q.loc[k41, "p90"] == q0.loc[k41, "p90"]
    assert q.loc[k42, "p90"] > q0.loc[k42, "p90"]


def test_gbm_learns_spread_dependent_width():
    df = synth(500)
    train, test = df[df["operating_date"] < df["operating_date"].iloc[0] + pd.Timedelta(days=400)], None
    test = df.drop(train.index)
    oof, pred = gbm_oof_and_test(train, test, ["spread", "hour_ending", "zone"])
    assert len(oof) == len(train) and len(pred) == len(test)
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
    m = test.merge(pred, on=["zone", "operating_date", "hour_ending"])
    width = m["p90"] - m["p10"]
    assert np.corrcoef(width, m["spread"])[0, 1] > 0.5
    cov = coverage(m[["p10", "p50", "p90"]].to_dict("records"), m["error"].tolist())
    assert cov == pytest.approx(0.8, abs=0.08)


def test_calibrated_gbm_hits_80pct_and_is_out_of_fold():
    df = synth(500)
    df["zone"] = df["zone"].astype(str)
    cut = df["operating_date"].iloc[0] + pd.Timedelta(days=400)
    train, test = df[df["operating_date"] < cut], df[df["operating_date"] >= cut]
    # a deliberately weak model (no spread feature) is miscalibrated; conformal calibration should fix coverage
    oof, pred = gbm_oof_and_test(train, test, ["hour_ending", "zone"], calibrate=True)
    m = test.merge(pred, on=["zone", "operating_date", "hour_ending"])
    cov = coverage(m[["p10", "p50", "p90"]].to_dict("records"), m["error"].tolist())
    assert cov == pytest.approx(0.8, abs=0.06)
    assert len(oof) == len(train) and (oof["p10"] <= oof["p90"]).all()
