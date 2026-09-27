"""Two-settlement battery backtest: turns predicted forecast uncertainty into dollars.

    revenue = sum_hours DA_award * DA_price
            + sum_15min (actual_output - DA_award) * RT_price * 0.25h
            - charging cost - degradation cost

Fleet: N Base Core units (~20 kW / 39.2 kWh each, ~85% round-trip), price taker.

Policy (same for every predictor, only the commitment fraction f differs):
  * Day-ahead: commit f * P MW in the two highest-priced DA hours of the day (all policies share this
    DA-price foresight simplification, so it cancels in comparisons).
  * Real time: the energy not committed is held back and discharged greedily in any 15-min interval
    (HE 7-22) whose RT price exceeds the day's highest DA price, up to the power not already committed.
  * Fixed rule: f = F0. Uncertainty-aware rule: f = clip(F0 * (W_ref / W)^ALPHA, F_MIN, 1), where W
    is the predicted p10-p90 width for the day and W_ref the zone's typical predicted width. Wide
    predicted uncertainty -> hold more for real time; calm -> commit more day-ahead.
Pre/post 2025-12-05 (RTC+B market redesign) results are reported separately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ercot_uncertainty.paths import ERCOT_DUCKDB

DUCKDB = str(ERCOT_DUCKDB)
ROOT = Path(__file__).resolve().parents[3]
RTC_B = date(2025, 12, 5)

# weather zone -> settlement point for prices
ZONE_SP = {
    "coast": "LZ_HOUSTON", "east": "LZ_NORTH", "far_west": "LZ_WEST", "north": "LZ_NORTH",
    "north_central": "LZ_NORTH", "south": "LZ_SOUTH", "south_central": "LZ_CPS", "west": "LZ_WEST",
    "system_total": "HB_HUBAVG",
}


@dataclass(frozen=True)
class Fleet:
    units: int = 1000
    unit_kw: float = 20.0
    unit_kwh: float = 39.2
    rte: float = 0.85
    degradation_usd_per_mwh: float = 10.0

    @property
    def power_mw(self) -> float:
        return self.units * self.unit_kw / 1000

    @property
    def energy_mwh(self) -> float:
        return self.units * self.unit_kwh / 1000


F0, F_MIN, ALPHA = 0.8, 0.2, 1.0


def settle(
    da_award_mw: np.ndarray,  # (24,) MW per hour-ending 1..24
    output_mw: np.ndarray,  # (96,) actual MW per 15-min interval
    da_price: np.ndarray,  # (24,) $/MWh
    rt_price: np.ndarray,  # (96,) $/MWh
    charge_mwh: float = 0.0,
    charge_price: float = 0.0,
    degradation_usd_per_mwh: float = 0.0,
) -> dict[str, float]:
    """Pure two-settlement accounting for one day."""
    award_15 = np.repeat(da_award_mw, 4)
    da_rev = float(np.sum(da_award_mw * da_price))
    rt_rev = float(np.sum((output_mw - award_15) * rt_price) * 0.25)
    discharged = float(np.sum(np.clip(output_mw, 0, None)) * 0.25)
    charge_cost = charge_mwh * charge_price
    degr = discharged * degradation_usd_per_mwh
    return {"da_rev": da_rev, "rt_rev": rt_rev, "charge_cost": charge_cost, "degradation": degr,
            "total": da_rev + rt_rev - charge_cost - degr, "discharged_mwh": discharged}


def simulate_day(f: float, da_price: np.ndarray, rt_price: np.ndarray, fleet: Fleet = Fleet()) -> dict[str, float]:
    """Apply the policy with commitment fraction f to one day's prices.

    Causal real-time rule: discharge in interval t only if the PREVIOUS interval's RT price cleared the
    threshold (no same-interval foresight). Profitability gate: never commit or discharge below the
    marginal cost of energy (overnight charging price / RTE + degradation).
    """
    P, E = fleet.power_mw, fleet.energy_mwh
    night = float(da_price[:6].mean())
    marginal_cost = night / fleet.rte + fleet.degradation_usd_per_mwh

    award = np.zeros(24)
    top = [h for h in np.argsort(-da_price)[:2] if da_price[h] > marginal_cost]
    committed_energy = min(f * P * len(top), f * E) if top else 0.0
    if top:
        award[top] = committed_energy / len(top)

    output = np.repeat(award, 4).astype(float)
    remaining = E - committed_energy
    threshold = max(float(da_price.max()), marginal_cost)
    for t in range(6 * 4, 22 * 4):  # HE 7..22
        if remaining <= 1e-9:
            break
        if rt_price[t - 1] > threshold:
            mw = min(P - output[t], remaining / 0.25)
            output[t] += mw
            remaining -= mw * 0.25
    discharged = E - remaining
    res = settle(award, output, da_price, rt_price, charge_mwh=discharged / fleet.rte, charge_price=night,
                 degradation_usd_per_mwh=fleet.degradation_usd_per_mwh)
    res["f"] = f
    return res


def commitment_fraction(width: float, ref_width: float) -> float:
    if not np.isfinite(width) or width <= 0:
        return F0
    return float(np.clip(F0 * (ref_width / width) ** ALPHA, F_MIN, 1.0))


def load_prices(sp: str, days: list[date]) -> dict[date, tuple[np.ndarray, np.ndarray]]:
    con = duckdb.connect(DUCKDB, read_only=True)
    lo, hi = min(days), max(days)
    da = con.execute(
        "select delivery_date, hour_ending, price_usd_per_mwh from da_spp_hourly "
        "where settlement_point=? and delivery_date between ? and ?", [sp, lo, hi]).fetchdf()
    rt = con.execute(
        "select delivery_date, hour_ending, interval_in_hour, price_usd_per_mwh from rt_spp_15min "
        "where settlement_point=? and delivery_date between ? and ?", [sp, lo, hi]).fetchdf()
    out = {}
    da_g = {d: g for d, g in da.groupby("delivery_date")}
    rt_g = {d: g for d, g in rt.groupby("delivery_date")}
    for d in days:
        key = pd.Timestamp(d)
        g1, g2 = da_g.get(key), rt_g.get(key)
        if g1 is None or g2 is None or len(g1) != 24 or len(g2) != 96:
            continue  # skip DST-change days (23/25 hours) and gaps
        out[d] = (g1.sort_values("hour_ending")["price_usd_per_mwh"].to_numpy(),
                  g2.sort_values(["hour_ending", "interval_in_hour"])["price_usd_per_mwh"].to_numpy())
    return out


def backtest(pred_widths: pd.DataFrame, ref_widths: dict[str, float], fleet: Fleet = Fleet()) -> pd.DataFrame:
    """pred_widths: zone, operating_date, width (predicted mean p90-p10 over peak hours)."""
    rows = []
    for zone, g in pred_widths.groupby("zone"):
        days = [pd.Timestamp(d).date() for d in g["operating_date"]]
        prices = load_prices(ZONE_SP[zone], days)
        for d, w in zip(days, g["width"]):
            if d not in prices:
                continue
            da, rt = prices[d]
            fixed = simulate_day(F0, da, rt, fleet)
            adapt = simulate_day(commitment_fraction(w, ref_widths[zone]), da, rt, fleet)
            rows.append({"zone": zone, "operating_date": d, "era": "post_rtcb" if d >= RTC_B else "pre_rtcb",
                         "f": adapt["f"], "fixed_usd": fixed["total"], "adaptive_usd": adapt["total"]})
    df = pd.DataFrame(rows)
    df["uplift_usd"] = df["adaptive_usd"] - df["fixed_usd"]
    return df


def widths_from_results(results: Path, examples: Path) -> pd.DataFrame:
    """Predicted widths for each example from an eval results file (falls back to baseline if unparsed)."""
    ex = {json.loads(l)["id"]: json.loads(l) for l in examples.open()}
    rows = []
    for line in results.open():
        r = json.loads(line)
        e = ex[r["id"]]
        w = r["width_mw"]
        if w is None:
            w = np.mean([v["p90"] - v["p10"] for v in e["baseline"].values()])
        rows.append({"zone": e["zone"], "operating_date": e["operating_date"], "width": w})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Dollar backtest of an uncertainty predictor")
    p.add_argument("--results", required=True, help="results/<split>__<backend>__<model>.jsonl")
    p.add_argument("--examples", default=str(ROOT / "data/examples/test.jsonl"))
    p.add_argument("--train-examples", default=str(ROOT / "data/examples/train.jsonl"))
    p.add_argument("--units", type=int, default=1000)
    a = p.parse_args()
    ref = {}
    for line in open(a.train_examples):
        e = json.loads(line)
        ref.setdefault(e["zone"], []).append(np.mean([v["p90"] - v["p10"] for v in e["baseline"].values()]))
    ref = {z: float(np.median(v)) for z, v in ref.items()}
    df = backtest(widths_from_results(Path(a.results), Path(a.examples)), ref, Fleet(units=a.units))
    summary = df.groupby("era")[["fixed_usd", "adaptive_usd", "uplift_usd"]].sum()
    summary["days"] = df.groupby("era").size()
    print(summary.round(0).to_string())
    out = Path(a.results).with_suffix(".dollars.csv")
    df.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
