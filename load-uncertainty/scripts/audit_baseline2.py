"""Round 2 of the numeric-baseline audit: feature engineering + training tweaks on top of the upgraded config
(500 trees, 15 leaves, per-zone conformal calibration). Two evaluation periods, both disjoint from test:
  A: train 2023-01..2025-08 -> val 2025-09..2026-02
  B: train 2023-01..2024-12 -> 2025-01-08..2025-08

    uv run python scripts/audit_baseline2.py
"""

import numpy as np
import pandas as pd

from ercot_uncertainty.baselines.numeric import QuantileGBM, time_blocks
from ercot_uncertainty.baselines.scoring import pinball
from ercot_uncertainty.features.build import FEATURES

Q = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
KEY = ["zone", "operating_date", "hour_ending"]
UPGRADE = dict(n_estimators=500, num_leaves=15, min_child_samples=80)

df = pd.read_parquet("data/parquet/hourly_features_ercot.parquet")
df = df[df.error.notna()].drop(columns=list(Q)).copy()
df = df.drop(columns=[c for c in df.columns if c.startswith("dev_")])  # rebuilt table already has them; recompute here
for c in ("is_holiday", "is_4cp_season"):
    df[c] = df[c].astype(int)
df["operating_date"] = pd.to_datetime(df["operating_date"])

# --- extra features -------------------------------------------------------------------------------------------------
lf = pd.read_parquet("data/parquet/lf_vintage.parquet", columns=["operating_date", "hour_ending", "zone", "model", "in_use",
                                                                  "forecast_mw", "vintage", "dst_repeated_hour"])
lf = lf[(lf.vintage == "dam") & ~lf.dst_repeated_hour]
lf["operating_date"] = pd.to_datetime(lf["operating_date"])
wide = lf.pivot_table(index=KEY, columns="model", values="forecast_mw")
dev = wide.sub(df.set_index(KEY)["fc_inuse"].reindex(wide.index), axis=0)
dev.columns = [f"dev_{m}" for m in dev.columns]
df = df.join(dev, on=KEY)
MODEL_DEV = list(dev.columns)

df["inuse_model_cat"] = df["inuse_model"].astype("category")
for c in ("fc_std", "fc_range", "inuse_minus_median", "revision"):
    df[f"{c}_rel"] = df[c] / df["fc_inuse"]
REL = ["fc_std_rel", "fc_range_rel", "inuse_minus_median_rel", "revision_rel"]

daily = df.groupby(["zone", "operating_date"])["error"].mean().rename("day_err").reset_index()
for lag in (2, 3):
    d = daily.copy()
    d["operating_date"] = d["operating_date"] + pd.Timedelta(days=lag)  # D-lag's mean peak error, known before cutoff
    df = df.merge(d.rename(columns={"day_err": f"day_err_lag{lag}"}), on=["zone", "operating_date"], how="left")
DAYLAG = ["day_err_lag2", "day_err_lag3"]

BASE_FEATS = [c for c in FEATURES if c in df and df[c].notna().any()] + [f"hist_{k}" for k in Q]


# --- fit / calibrate / score ------------------------------------------------------------------------------------------
def fit_predict(tr, te, feats, params, weight_halflife=None, seeds=(0,)):
    preds = []
    for s in seeds:
        m = QuantileGBM(feats, categorical=["zone", "inuse_model_cat"])
        m.params = {**m.params, **params, "random_state": s}
        w = None
        if weight_halflife:
            age = (tr.operating_date.max() - tr.operating_date).dt.days
            w = 0.5 ** (age / weight_halflife)
        x = m._x(tr)
        m.models = {k: __import__("lightgbm").LGBMRegressor(objective="quantile", alpha=t, **m.params).fit(x, tr.error, sample_weight=w)
                    for k, t in Q.items()}
        preds.append(m.predict(te)[list(Q)].to_numpy())
    return pd.DataFrame(np.mean(preds, axis=0), columns=list(Q), index=te.index)


def run(tr, te, feats, params=UPGRADE, weight_halflife=None, seeds=(0,)):
    p = fit_predict(tr, te, feats, params, weight_halflife, seeds)
    # per-zone conformal widening from 3-fold out-of-fold predictions on train
    b = time_blocks(tr.operating_date, 3)
    oof = pd.concat([fit_predict(tr[b != k], tr[b == k], feats, params, weight_halflife, seeds[:1]) for k in range(3)]).loc[tr.index]
    nc = np.maximum(oof.p10 - tr.error, tr.error - oof.p90)
    adj = nc.groupby(tr.zone).quantile(0.8)
    a = te.zone.map(adj).to_numpy()
    p["p10"] -= a
    p["p90"] += a
    p = pd.DataFrame(np.sort(p.to_numpy(), axis=1), columns=list(Q), index=te.index)
    loss = np.mean([pinball(te.error.to_numpy(), p[k].to_numpy(), t).mean() for k, t in Q.items()])
    cov = ((te.error >= p.p10) & (te.error <= p.p90)).mean()
    return loss, cov


periods = {
    "A": (df[df.split == "train"], df[df.split == "val"]),
    "B": (df[(df.split == "train") & (df.operating_date < "2025-01-01")],
          df[(df.split == "train") & (df.operating_date >= "2025-01-08")]),
}
experiments = {
    "upgrade (500 trees, 15 leaves + calibration)": dict(feats=BASE_FEATS),
    "+ in-use model identity": dict(feats=BASE_FEATS + ["inuse_model_cat"]),
    "+ each model's deviation from in-use": dict(feats=BASE_FEATS + MODEL_DEV),
    "+ relative (per-MW) features": dict(feats=BASE_FEATS + REL),
    "+ recent daily errors (D-2, D-3)": dict(feats=BASE_FEATS + DAYLAG),
    "+ recency weighting (1-yr half-life)": dict(feats=BASE_FEATS, weight_halflife=365),
    "+ average of 3 seeds": dict(feats=BASE_FEATS, seeds=(0, 1, 2)),
}
rows = []
for name, e in experiments.items():
    r = {"experiment": name}
    for pn, (tr, te) in periods.items():
        loss, cov = run(tr, te, e["feats"], weight_halflife=e.get("weight_halflife"), seeds=e.get("seeds", (0,)))
        r[f"{pn}_MW"], r[f"{pn}_cov"] = loss, cov
    rows.append(r)
    print(f"done: {name}", flush=True)
out = pd.DataFrame(rows).set_index("experiment")
for pn in periods:
    out[f"{pn}_vs_upgrade_%"] = 100 * (out[f"{pn}_MW"] / out.iloc[0][f"{pn}_MW"] - 1)
out["avg_%"] = out[[f"{pn}_vs_upgrade_%" for pn in periods]].mean(axis=1)
print(out[["A_MW", "A_cov", "A_vs_upgrade_%", "B_MW", "B_cov", "B_vs_upgrade_%", "avg_%"]].round(3).to_string())
out.to_csv("outputs/baseline_audit_round2.csv")
