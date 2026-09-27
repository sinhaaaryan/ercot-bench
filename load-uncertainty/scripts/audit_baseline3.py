"""Round 3 of the numeric-baseline audit: (a) algorithms beyond LightGBM, (b) more inputs.

Same two periods as round 2 (both disjoint from test). Algorithms are compared WITHOUT the conformal step so the
comparison is like-for-like; inputs are compared with LightGBM (500 trees, 15 leaves) + per-zone calibration.

    uv run python scripts/audit_baseline3.py
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ercot_uncertainty.baselines.numeric import time_blocks
from ercot_uncertainty.baselines.scoring import pinball
from ercot_uncertainty.features.build import FEATURES

Q = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
KEY = ["zone", "operating_date", "hour_ending"]
LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, verbose=-1)

df = pd.read_parquet("data/parquet/hourly_features_ercot.parquet")
df = df[df.error.notna()].drop(columns=list(Q)).copy()
df = df.drop(columns=[c for c in df.columns if c.startswith("dev_")])  # rebuilt table already has them; recompute here
for c in ("is_holiday", "is_4cp_season"):
    df[c] = df[c].astype(int)
df["operating_date"] = pd.to_datetime(df["operating_date"])

# --- round-2 winners: per-model deviations from the in-use forecast + in-use identity ----------------------------------
lf = pd.read_parquet("data/parquet/lf_vintage.parquet", columns=["operating_date", "hour_ending", "zone", "model", "forecast_mw",
                                                                  "vintage", "dst_repeated_hour"])
lf = lf[~lf.dst_repeated_hour]
lf["operating_date"] = pd.to_datetime(lf["operating_date"])
dam = lf[lf.vintage == "dam"].pivot_table(index=KEY, columns="model", values="forecast_mw")
prev = lf[lf.vintage == "prev"].pivot_table(index=KEY, columns="model", values="forecast_mw")
models = list(dam.columns)
inuse = df.set_index(KEY)["fc_inuse"]
for m in models:
    df[f"dev_{m}"] = (dam[m] - inuse.reindex(dam.index)).reindex(df.set_index(KEY).index).to_numpy()
df["inuse_model_cat"] = df["inuse_model"].astype("category")
R2 = [f"dev_{m}" for m in models] + ["inuse_model_cat"]

# --- new inputs --------------------------------------------------------------------------------------------------------
# (1) each model's recent track record: bias and MAE of its own forecast over the last 7 / 30 days, days <= D-2
act = pd.read_parquet("data/parquet/actual_load.parquet")
act = act[~act.dst_repeated_hour & act.hour_ending.isin(df.hour_ending.unique())]
act["operating_date"] = pd.to_datetime(act["operating_date"])
a = act.set_index(KEY)["actual_mw"]
merr = dam.reindex(a.index).rsub(a, axis=0)  # actual - model forecast, peak hours
daily = merr.groupby(level=["zone", "operating_date"]).mean()  # per zone-day mean over peak hours
TRACK = []
for m in models:
    s = daily[m].unstack("zone").asfreq("D")  # calendar-regular so shifts are in days
    lag = s.shift(2)  # known by the D-1 10:00 cutoff: D-2
    for name, frame in {f"trk_bias7_{m}": lag.rolling(7, min_periods=4).mean(),
                        f"trk_mae30_{m}": lag.abs().rolling(30, min_periods=15).mean()}.items():
        long = frame.stack().rename(name).reset_index()
        df = df.merge(long, on=["operating_date", "zone"], how="left")
        TRACK.append(name)
# the in-use model's own track record, and how it ranks vs the best recent model
df["trk_inuse_mae30"] = [row[f"trk_mae30_{m}"] if isinstance(m, str) and f"trk_mae30_{m}" in df else np.nan
                         for m, row in zip(df["inuse_model"], df[[f"trk_mae30_{x}" for x in models]].to_dict("records"))]
df["trk_best_mae30"] = df[[f"trk_mae30_{x}" for x in models]].min(axis=1)
df["trk_inuse_minus_best"] = df["trk_inuse_mae30"] - df["trk_best_mae30"]
TRACK_SUMMARY = ["trk_inuse_mae30", "trk_best_mae30", "trk_inuse_minus_best"]
# (2) each model's revision since the prior-day post
REV = []
for m in models:
    df[f"rev_{m}"] = (dam[m] - prev[m]).reindex(df.set_index(KEY).index).to_numpy()
    REV.append(f"rev_{m}")
# (3) heat index proxy (humid heat drives AC load)
t_f = df["temp_c"] * 9 / 5 + 32
td_f = df["dewpoint_c"] * 9 / 5 + 32
df["heat_index_proxy"] = t_f + 0.5 * (td_f - 55).clip(lower=0)
HEAT = ["heat_index_proxy"]

BASE = [c for c in FEATURES if c in df and df[c].notna().any()] + [f"hist_{k}" for k in Q] + R2
CATS = ["zone", "inuse_model_cat"]


def X(d, feats):
    x = d[feats].copy()
    for c in CATS:
        if c in x:
            x[c] = x[c].astype("category")
    return x


def score(p, te):
    p = pd.DataFrame(np.sort(np.asarray(p), axis=1), columns=list(Q), index=te.index)
    loss = np.mean([pinball(te.error.to_numpy(), p[k].to_numpy(), t).mean() for k, t in Q.items()])
    return loss, ((te.error >= p.p10) & (te.error <= p.p90)).mean(), p


# ---- algorithms ----------------------------------------------------------------------------------------------------
def algo_lgb(tr, te, feats):
    x, xt = X(tr, feats), X(te, feats)
    return np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=t, **LGB).fit(x, tr.error).predict(xt) for t in Q.values()])


def algo_catboost(tr, te, feats):
    from catboost import CatBoostRegressor, Pool
    cats = [c for c in CATS if c in feats]
    to_pool = lambda d, y=None: Pool(d[feats].assign(**{c: d[c].astype(str) for c in cats}), y, cat_features=cats)
    m = CatBoostRegressor(loss_function="MultiQuantile:alpha=0.1,0.5,0.9", iterations=1500, learning_rate=0.05, depth=6,
                          verbose=0, thread_count=12, random_seed=0)
    m.fit(to_pool(tr, tr.error))
    return m.predict(to_pool(te))


def algo_xgb(tr, te, feats):
    import xgboost as xgb
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(list(Q.values())), n_estimators=600,
                         learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                         tree_method="hist", enable_categorical=True, n_jobs=12)
    m.fit(X(tr, feats), tr.error)
    return m.predict(X(te, feats))


def algo_qrf(tr, te, feats):
    from quantile_forest import RandomForestQuantileRegressor
    num = [c for c in feats if c not in CATS]
    enc = lambda d: pd.concat([d[num].fillna(-9999), pd.get_dummies(d[[c for c in CATS if c in feats]].astype(str))], axis=1)
    xtr, xte = enc(tr), enc(te).reindex(columns=enc(tr).columns, fill_value=0)
    m = RandomForestQuantileRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=12, random_state=0)
    m.fit(xtr, tr.error)
    return m.predict(xte, quantiles=list(Q.values()))


def algo_two_stage(tr, te, feats):
    """Median model + model of |residual| size; quantiles = median + z * size with z from train out-of-fold residuals."""
    x, xt = X(tr, feats), X(te, feats)
    med = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **LGB).fit(x, tr.error)
    b = time_blocks(tr.operating_date, 3)
    oof = np.zeros(len(tr))
    for k in range(3):
        mk = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **LGB).fit(x[b != k], tr.error[b != k])
        oof[b == k] = mk.predict(x[b == k])
    resid = tr.error.to_numpy() - oof
    size = lgb.LGBMRegressor(objective="regression_l1", **LGB).fit(x, np.abs(resid))
    s_tr = np.clip(size.predict(x), 1, None)
    z = np.quantile(resid / s_tr, [0.1, 0.9])
    m, s = med.predict(xt), np.clip(size.predict(xt), 1, None)
    return np.column_stack([m + z[0] * s, m, m + z[1] * s])


periods = {
    "A": (df[df.split == "train"], df[df.split == "val"]),
    "B": (df[(df.split == "train") & (df.operating_date < "2025-01-01")], df[(df.split == "train") & (df.operating_date >= "2025-01-08")]),
}
algos = {"LightGBM": algo_lgb, "CatBoost MultiQuantile": algo_catboost, "XGBoost quantile": algo_xgb,
         "Quantile regression forest": algo_qrf, "Two-stage (median + size)": algo_two_stage}
rows, preds = [], {}
for name, f in algos.items():
    r = {"algorithm": name}
    for pn, (tr, te) in periods.items():
        t0 = time.time()
        p = f(tr, te, BASE)
        loss, cov, preds[(name, pn)] = score(p, te)
        r[f"{pn}_MW"], r[f"{pn}_cov"], r[f"{pn}_sec"] = loss, cov, time.time() - t0
    rows.append(r)
    print("done", name, flush=True)
for other in ("CatBoost MultiQuantile", "XGBoost quantile", "Quantile regression forest"):
    r = {"algorithm": f"Average(LightGBM, {other})"}
    for pn, (tr, te) in periods.items():
        loss, cov, _ = score((preds[("LightGBM", pn)] + preds[(other, pn)]) / 2, te)
        r[f"{pn}_MW"], r[f"{pn}_cov"] = loss, cov
    rows.append(r)
out = pd.DataFrame(rows).set_index("algorithm")
for pn in periods:
    out[f"{pn}_vs_lgb_%"] = 100 * (out[f"{pn}_MW"] / out.loc["LightGBM", f"{pn}_MW"] - 1)
out["avg_%"] = out[[f"{pn}_vs_lgb_%" for pn in periods]].mean(axis=1)
print("\n=== (a) algorithms, same inputs (round-2 set), before calibration")
print(out[["A_MW", "A_cov", "A_vs_lgb_%", "B_MW", "B_cov", "B_vs_lgb_%", "avg_%", "A_sec"]].round(3).to_string())


# ---- inputs (LightGBM + per-zone calibration) --------------------------------------------------------------------------
def lgb_cal(tr, te, feats):
    p = pd.DataFrame(algo_lgb(tr, te, feats), columns=list(Q), index=te.index)
    b = time_blocks(tr.operating_date, 3)
    oof = pd.concat([pd.DataFrame(algo_lgb(tr[b != k], tr[b == k], feats), columns=list(Q), index=tr[b == k].index)
                     for k in range(3)]).loc[tr.index]
    adj = np.maximum(oof.p10 - tr.error, tr.error - oof.p90).groupby(tr.zone).quantile(0.8)
    a = te.zone.map(adj).to_numpy()
    p["p10"] -= a
    p["p90"] += a
    return score(p, te)[:2]


inputs = {"round-2 set (deviations + in-use identity)": BASE,
          "+ each model's recent track record": BASE + TRACK + TRACK_SUMMARY,
          "+ track-record summary only (in-use vs best)": BASE + TRACK_SUMMARY,
          "+ each model's revision": BASE + REV,
          "+ heat index": BASE + HEAT}
rows = []
for name, feats in inputs.items():
    r = {"inputs": name}
    for pn, (tr, te) in periods.items():
        r[f"{pn}_MW"], r[f"{pn}_cov"] = lgb_cal(tr, te, feats)
    rows.append(r)
    print("done", name, flush=True)
out2 = pd.DataFrame(rows).set_index("inputs")
for pn in periods:
    out2[f"{pn}_vs_r2_%"] = 100 * (out2[f"{pn}_MW"] / out2.iloc[0][f"{pn}_MW"] - 1)
out2["avg_%"] = out2[[f"{pn}_vs_r2_%" for pn in periods]].mean(axis=1)
print("\n=== (b) inputs, LightGBM + calibration")
print(out2.round(3).to_string())
out.to_csv("outputs/baseline_audit_round3_algos.csv")
out2.to_csv("outputs/baseline_audit_round3_inputs.csv")
