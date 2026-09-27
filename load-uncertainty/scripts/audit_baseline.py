"""Audit the numeric (non-LLM) baseline: is LightGBM-quantile the best non-LLM uncertainty model we can build?

Selection uses VALIDATION only (test stays untouched for the final report). All candidates are fit on train.
Prints: feature importance of the current model, then val pinball / coverage / width for each candidate.

    uv run python scripts/audit_baseline.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from ercot_uncertainty.baselines.numeric import QuantileGBM
from ercot_uncertainty.baselines.scoring import pinball
from ercot_uncertainty.features.build import FEATURES

Q = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
df = pd.read_parquet("data/parquet/hourly_features_ercot.parquet")
df = df[df["error"].notna()].drop(columns=list(Q)).copy()
df["is_holiday"] = df["is_holiday"].astype(int)
df["is_4cp_season"] = df["is_4cp_season"].astype(int)
feats = [c for c in FEATURES if c in df and df[c].notna().any()] + [f"hist_{k}" for k in Q]
tr, va = df[df.split == "train"].copy(), df[df.split == "val"].copy()


def score(pred: pd.DataFrame, d: pd.DataFrame) -> dict:
    loss = np.mean([pinball(d["error"].to_numpy(), pred[k].to_numpy(), t).mean() for k, t in Q.items()])
    cov = ((d["error"].to_numpy() >= pred["p10"].to_numpy()) & (d["error"].to_numpy() <= pred["p90"].to_numpy())).mean()
    return {"val_pinball_MW": loss, "coverage_80": cov, "width_MW": (pred["p90"] - pred["p10"]).mean()}


def sort_q(p: pd.DataFrame) -> pd.DataFrame:
    a = np.sort(p[list(Q)].to_numpy(), axis=1)
    return pd.DataFrame(a, columns=list(Q), index=p.index)


results = {}

# 0. what the pipeline uses today (fit on train, predicted on val by build.py)
results["LightGBM (current: 250 trees, 7 leaves)"] = score(va[[f"gbm_{k}" for k in Q]].set_axis(list(Q), axis=1), va)

# 1. simple heuristics
clim = tr.groupby(["zone", "hour_ending"])["error"].quantile(list(Q.values())).unstack()
clim.columns = list(Q)
results["Same range every day (zone/hour)"] = score(va.join(clim, on=["zone", "hour_ending"])[list(Q)], va)
h = va[[f"hist_{k}" for k in Q]].set_axis(list(Q), axis=1)
results["Recent 90-day errors (zone/hour)"] = score(h.fillna(h.mean()), va)

# 2. scale heuristic: standardized errors x a day-specific scale (ERCOT model spread + recent MAE)
for name, scale_cols in {"spread": ["fc_std"], "spread+recentMAE": ["fc_std", "err_mae30"]}.items():
    def scale(d):
        s = sum(d[c].fillna(d[c].median()) / tr[c].median() for c in scale_cols) / len(scale_cols)
        return s.clip(lower=0.2)
    z = tr["error"] / scale(tr)
    zq = pd.DataFrame({"z": z, "zone": tr["zone"]}).groupby("zone")["z"].quantile(list(Q.values())).unstack()
    zq.columns = list(Q)
    p = va[["zone"]].join(zq, on="zone")[list(Q)].mul(scale(va), axis=0)
    results[f"Heuristic: typical range x {name}"] = score(p, va)

# 3. LightGBM variants (hyperparameters, more trees, per-zone normalization)
grid = {
    "LightGBM 500 trees, 15 leaves": dict(n_estimators=500, num_leaves=15, min_child_samples=80),
    "LightGBM 1000 trees, 7 leaves, lr .02": dict(n_estimators=1000, num_leaves=7, min_child_samples=80, learning_rate=0.02),
    "LightGBM 400 trees, 31 leaves, min_child 200": dict(n_estimators=400, num_leaves=31, min_child_samples=200),
}
for name, params in grid.items():
    m = QuantileGBM(feats)
    m.params = {**m.params, **params}
    results[name] = score(sort_q(m.fit(tr).predict(va)[list(Q)]), va)

# 3b. target normalized by the zone's typical error size, so trees share structure across zones
zs = tr.groupby("zone")["error"].apply(lambda s: s.abs().mean())
trn, van = tr.copy(), va.copy()
trn["error"] = tr["error"] / tr["zone"].map(zs)
m = QuantileGBM(feats).fit(trn)
p = m.predict(van)[list(Q)].mul(va["zone"].map(zs).to_numpy(), axis=0)
results["LightGBM on zone-normalized error"] = score(sort_q(p), va)

# 4. conformal calibration of the current model's interval (CQR) using out-of-fold train predictions
oof = tr[[f"gbm_{k}" for k in Q]].set_axis(list(Q), axis=1)
nonconf = np.maximum(oof["p10"] - tr["error"], tr["error"] - oof["p90"])
adj = tr.assign(nc=nonconf).groupby("zone")["nc"].quantile(0.8)
cur = va[[f"gbm_{k}" for k in Q]].set_axis(list(Q), axis=1).copy()
a = va["zone"].map(adj).to_numpy()
cur["p10"] -= a
cur["p90"] += a
results["LightGBM + per-zone conformal calibration"] = score(sort_q(cur), va)

# 5. simple ensemble: average of LightGBM and 90-day quantiles
ens = (va[[f"gbm_{k}" for k in Q]].set_axis(list(Q), axis=1) + h.fillna(h.mean())) / 2
results["Average(LightGBM, 90-day)"] = score(sort_q(ens), va)

out = pd.DataFrame(results).T.sort_values("val_pinball_MW")
out["vs_current_%"] = 100 * (out["val_pinball_MW"] / out.loc["LightGBM (current: 250 trees, 7 leaves)", "val_pinball_MW"] - 1)
print(f"validation: {len(va)} zone-hours ({va.operating_date.nunique()} days, incl. system_total)\n")
print(out.round(3).to_string())

# feature importance (gain) of the current p90 and p10 models, fit on train
m = QuantileGBM(feats).fit(tr)
imp = pd.DataFrame({k: pd.Series(m.models[k].booster_.feature_importance("gain"), index=feats) for k in ("p10", "p90")})
imp = (imp / imp.sum()).mean(axis=1).sort_values(ascending=False)
print("\nWhat the current model leans on (share of total split gain, p10+p90 models):")
print("  " + ", ".join(f"{k} {v:.0%}" for k, v in imp.head(14).items()))
