"""Retail load-hedging translation: what a better median (p50) of ERCOT's forecast error is worth to a load-serving
entity (Base is also a retail electricity provider).

Setup: a retailer buys its expected load day-ahead. Baseline: it buys ERCOT's day-ahead forecast (error model = 0).
With a model: it buys ERCOT's forecast + the model's predicted median error (p50). Whatever it misses is settled in
real time:  imbalance P&L per interval = (actual - purchased) x (RT price - DA price)  (positive = cost).
Reported per model on the evening peak (HE16-20), per zone, scaled to a 1 GW-average portfolio in that zone:
  - imbalance MWh (volume of energy settled at the volatile RT price),
  - signed imbalance cost, and |imbalance| x |RT - DA| as a risk measure.
Price-taker; no transaction costs; zonal load share assumed proportional.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ercot_uncertainty.paths import ERCOT_DUCKDB

from ercot_uncertainty.eval_report import ZONE_SP, _long, _meta
from ercot_uncertainty.eval import load_examples

DUCKDB = str(ERCOT_DUCKDB)


def _prices(days: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DUCKDB, read_only=True)
    lo, hi = min(days), max(days)
    da = con.execute("select settlement_point as sp, cast(delivery_date as varchar) as dday, hour_ending as hour, "
                     "price_usd_per_mwh as da from da_spp_hourly where not dst_repeated_hour and delivery_date between ? and ?",
                     [lo, hi]).fetchdf()
    rt = con.execute("select settlement_point as sp, cast(delivery_date as varchar) as dday, hour_ending as hour, "
                     "avg(price_usd_per_mwh) as rt from rt_spp_15min where not dst_repeated_hour and delivery_date between ? and ? "
                     "group by 1, 2, 3", [lo, hi]).fetchdf()
    return da.merge(rt, on=["sp", "dday", "hour"])


def imbalance(split: str, files: list[Path]) -> pd.DataFrame:
    examples = {e["id"]: e for e in load_examples(split)}
    meta = _meta(examples, examples)
    runs = {f.stem: {json.loads(l)["id"]: json.loads(l) for l in f.open()} for f in files}
    ids = [i for i in examples if all(i in r for r in runs.values())]
    px = _prices(sorted({examples[i]["operating_date"] for i in ids})).set_index(["sp", "dday", "hour"])
    # zone load level (MW) to express a 1 GW-average portfolio: scale MW errors by 1000 / zone mean load
    fc = pd.read_parquet("data/parquet/hourly_features_ercot.parquet", columns=["zone", "operating_date", "hour_ending", "fc_inuse"])
    zone_load = fc.groupby("zone")["fc_inuse"].mean()
    out = []
    for name, r in runs.items():
        d = _long({i: r[i] for i in ids}, examples).join(meta[["zone", "day"]], on="id")
        key = list(zip(d.zone.map(ZONE_SP), d.day, d.hour))
        p = px.reindex(key)
        spread = (p["rt"] - p["da"]).to_numpy()
        k = 1000.0 / d.zone.map(zone_load).to_numpy()  # MW -> MW of a 1 GW portfolio in that zone
        miss_model = (d.y - d.p50).to_numpy() * k       # actual - (ERCOT forecast + predicted median error)
        miss_ercot = d.y.to_numpy() * k
        ok = np.isfinite(spread)
        out.append({"run": name, "zone_hours": int(ok.sum()),
                    "imbalance_MWh_model": np.abs(miss_model[ok]).sum(), "imbalance_MWh_ercot": np.abs(miss_ercot[ok]).sum(),
                    "signed_cost_model_$": (miss_model[ok] * spread[ok]).sum(), "signed_cost_ercot_$": (miss_ercot[ok] * spread[ok]).sum(),
                    "risk_model_$": (np.abs(miss_model[ok]) * np.abs(spread[ok])).sum(),
                    "risk_ercot_$": (np.abs(miss_ercot[ok]) * np.abs(spread[ok])).sum()})
    df = pd.DataFrame(out)
    df["imbalance_vs_ERCOT_%"] = 100 * (df.imbalance_MWh_model / df.imbalance_MWh_ercot - 1)
    df["risk_vs_ERCOT_%"] = 100 * (df["risk_model_$"] / df["risk_ercot_$"] - 1)
    return df.sort_values("imbalance_MWh_model")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(imbalance(a.split, [Path(f) for f in a.results]).round(1).to_string(index=False))
