"""Report card for the uncertainty eval: every run scored the same way, against one declared reference.

    uv run python -m ercot_uncertainty.eval_report --split test_sample400 \
        --reference results/test_sample400__baseline__lightgbm-v2.jsonl results/test_sample400__*.jsonl \
        [--dollars] [--out results/report_test_sample400.md]

What it reports (see docs/EVAL.md for the rationale):
  1. Headline: pinball loss (MW) over all zones incl. system total, AND a zone-normalized skill score
     (each of the 8 weather zones' improvement vs the reference, averaged with equal weight), plus the
     system total on its own. Zone-days on the same date share weather/text, so every confidence interval is a
     bootstrap over operating DAYS.
  2. Calibration of each percentile (share of actuals below p10/p50/p90, targets 10/50/90%), overall and by zone,
     plus a sigma view (reading the p10-p90 range as +/-1.28 sigma): within 1/2 sigma, beyond 3 sigma.
  3. Parse failures: one rule everywhere -- an unparseable answer is scored as "trust ERCOT exactly"
     (p10 = p50 = p90 = 0 MW error), and the failure rate is reported separately.
  4. Thesis slices: days ERCOT's models agreed / disagreed, big-miss days, discussion themes (storms, severe,
     heat, heavy rain, tropical, quiet), and month.
  5. Optional: dollar backtest (sim/settlement.py) of the commit-less-when-uncertain rule vs a fixed rule.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ercot_uncertainty.eval import load_examples
from ercot_uncertainty.paths import ERCOT_DUCKDB
from ercot_uncertainty.task import parse_percentiles

Q = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
Z90 = 1.2816  # p90 of a standard normal: the p10-p90 range spans +/-1.28 sigma if the errors were Gaussian
THEMES = {
    "storms": r"thunderstorm|\btstorms?\b|\bstorms?\b|convecti",
    "severe": r"severe|tornado|\bhail\b|damaging wind|\bshear\b",
    "heat": r"heat advisory|heat index|excessive heat|record high|triple.digit",
    "heavy_rain": r"flash flood|flooding|heavy rain|excessive rain",
    "tropical": r"tropical|hurricane",
    "quiet": r"\bquiet\b|clear skies|little change|benign",
}


def _long(run: dict[str, dict], examples: dict[str, dict]) -> pd.DataFrame:
    """One row per (example, hour) with the run's percentiles (fallback: zero-width at 0 when unparseable)."""
    rows = []
    for i, r in run.items():
        ex = examples[i]
        hours = [int(h) for h in ex["actual_error"]]
        p = parse_percentiles(r["completion"], hours)
        for h in hours:
            q = p.hours[h] if p.ok else {"p10": 0.0, "p50": 0.0, "p90": 0.0}
            rows.append({"id": i, "hour": h, "y": ex["actual_error"][str(h)], "parse_ok": p.ok, **q})
    d = pd.DataFrame(rows)
    d["loss"] = np.mean([np.maximum(t * (d.y - d[k]), (t - 1) * (d.y - d[k])) for k, t in Q.items()], axis=0)
    return d


def _meta(examples: dict[str, dict], text_source: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for i, ex in examples.items():
        prompt = text_source.get(i, ex)["prompt"][1]["content"].lower()
        rows.append({"id": i, "zone": ex["zone"], "day": ex["operating_date"], "month": ex["operating_date"][:7],
                     "models_agreed": bool(ex["meta"].get("models_agreed")), "big_miss": bool(ex["meta"].get("big_miss")),
                     **{f"theme_{k}": bool(re.search(v, prompt)) for k, v in THEMES.items()}})
    return pd.DataFrame(rows).set_index("id")


def _day_boot(ids_days: np.ndarray, n: int = 2000, seed: int = 0) -> list[np.ndarray]:
    uniq = np.unique(ids_days)
    by_day = [np.where(ids_days == d)[0] for d in uniq]
    rng = np.random.default_rng(seed)
    return [np.concatenate([by_day[j] for j in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n)]


ZONE_SP = {"coast": "LZ_HOUSTON", "east": "LZ_NORTH", "far_west": "LZ_WEST", "north": "LZ_NORTH",
           "north_central": "LZ_NORTH", "south": "LZ_SOUTH", "south_central": "LZ_CPS", "west": "LZ_WEST",
           "system_total": "HB_HUBAVG"}


def _da_prices(meta: pd.DataFrame) -> dict:
    """(zone, date, hour_ending) -> day-ahead settlement point price ($/MWh), read-only from the teammate's DuckDB."""
    import duckdb
    con = duckdb.connect(str(ERCOT_DUCKDB), read_only=True)
    lo, hi = meta["day"].min(), meta["day"].max()
    df = con.execute("select settlement_point, delivery_date, hour_ending, price_usd_per_mwh from da_spp_hourly "
                     "where not dst_repeated_hour and delivery_date between ? and ?", [lo, hi]).fetchdf()
    by_sp = {(r.settlement_point, str(r.delivery_date)[:10], int(r.hour_ending)): r.price_usd_per_mwh for r in df.itertuples()}
    return {(z, d, h): by_sp.get((ZONE_SP[z], d, h)) for z in ZONE_SP for d in meta["day"].unique() for h in range(16, 21)}


def _climatology(train_split: str = "train") -> dict:
    """'Same range every day': train-period p10/p50/p90 of the error per (zone, hour)."""
    errs = {}
    for e in load_examples(train_split):
        for h, y in e["actual_error"].items():
            errs.setdefault((e["zone"], int(h)), []).append(y)
    return {k: dict(zip(Q, np.quantile(v, list(Q.values())))) for k, v in errs.items()}


def _extra_metrics(long: dict, meta: pd.DataFrame, ref_name: str, ids: list[str], boots: list[np.ndarray]) -> pd.DataFrame:
    """Price-weighted pinball, median accuracy vs ERCOT's point forecast, skill vs climatology, per run."""
    prices = _da_prices(meta)
    clim = _climatology()
    ref = long[ref_name]
    rows = []
    idx_of = {i: k for k, i in enumerate(ids)}
    for name, d in long.items():
        d = d.join(meta[["zone", "day"]], on="id")
        w = np.array([prices.get((z, dd, h)) or np.nan for z, dd, h in zip(d.zone, d.day, d.hour)], dtype=float)
        w = np.where(np.isfinite(w), np.clip(w, 0, None), np.nanmedian(w))
        c = np.array([[clim[(z, h)][k] for k in Q] for z, h in zip(d.zone, d.hour)])
        clim_loss = np.mean([np.maximum(t * (d.y - c[:, j]), (t - 1) * (d.y - c[:, j])) for j, t in enumerate(Q.values())], axis=0)
        pw = float(np.sum(w * d.loss) / np.sum(w))
        pw_ref = float(np.sum(w * ref.loss.to_numpy()) / np.sum(w))
        # per-example weighted deltas for a day-clustered CI
        ex = pd.DataFrame({"id": d.id, "wl": w * d.loss, "wr": w * ref.loss.to_numpy(), "w": w}).groupby("id").sum().loc[ids]
        bd = np.array([(ex.wl.to_numpy()[b].sum() - ex.wr.to_numpy()[b].sum()) / ex.w.to_numpy()[b].sum() for b in boots])
        mae_model = float(np.mean(np.abs(d.y - d.p50)))
        mae_ercot = float(np.mean(np.abs(d.y)))
        rows.append({"run": name,
                     "price_weighted_pinball": pw, "price_weighted_vs_ref_%": 100 * (pw / pw_ref - 1),
                     "pw_ci95_%": f"[{100 * np.percentile(bd, 2.5) / pw_ref:+.1f}, {100 * np.percentile(bd, 97.5) / pw_ref:+.1f}]",
                     "p50_MAE_MW": mae_model, "ERCOT_MAE_MW": mae_ercot,
                     "p50_vs_ERCOT_%": 100 * (mae_model / mae_ercot - 1),
                     "skill_vs_same_range_every_day_%": 100 * (1 - d.loss.mean() / clim_loss.mean())})
    return pd.DataFrame(rows).sort_values("price_weighted_pinball")


def build(split: str, files: list[Path], reference: Path) -> dict:
    examples = {e["id"]: e for e in load_examples(split)}
    full_text = examples if "numeric" not in split else {}
    meta = _meta(examples, full_text)
    runs = {f.stem: {json.loads(l)["id"]: json.loads(l) for l in f.open()} for f in [reference] + [f for f in files if f != reference]}
    ids = [i for i in examples if all(i in r for r in runs.values())]
    ref_name = reference.stem
    per_ex = {}   # run -> DataFrame indexed by id: mean loss per example + calibration counts
    long = {}
    for name, r in runs.items():
        d = _long({i: r[i] for i in ids}, examples)
        long[name] = d
        g = d.groupby("id")
        per_ex[name] = pd.DataFrame({
            "loss": g.loss.mean(), "parse_ok": g.parse_ok.first(),
            "below_p10": g.apply(lambda x: np.mean(x.y < x.p10)), "below_p50": g.apply(lambda x: np.mean(x.y < x.p50)),
            "below_p90": g.apply(lambda x: np.mean(x.y < x.p90)),
            "width": g.apply(lambda x: np.mean(x.p90 - x.p10)),
        }).loc[ids].join(meta)
    days = per_ex[ref_name]["day"].to_numpy()
    boots = _day_boot(days)
    ref = per_ex[ref_name]
    zones = [z for z in ref.zone.unique() if z != "system_total"]

    def zone_skill(df_run, idx):
        s = []
        for z in zones:
            m = (ref.zone.to_numpy()[idx] == z)
            if m.sum() == 0:
                continue
            s.append(1 - df_run.loss.to_numpy()[idx][m].mean() / ref.loss.to_numpy()[idx][m].mean())
        return 100 * float(np.mean(s))

    headline = []
    for name, d in per_ex.items():
        diff = d.loss.to_numpy() - ref.loss.to_numpy()
        bd = np.array([diff[b].mean() for b in boots])
        allidx = np.arange(len(ids))
        zs = zone_skill(d, allidx)
        bz = np.array([zone_skill(d, b) for b in boots[:500]])
        sysm = (d.zone == "system_total").to_numpy()
        sys_d = 100 * (1 - d.loss[sysm].mean() / ref.loss[sysm].mean()) if sysm.any() else np.nan
        lp = long[name]
        zsig = (lp.y - lp.p50) / ((lp.p90 - lp.p10).clip(lower=1e-9) / (2 * Z90))
        headline.append({
            "run": name, "n": len(ids), "days": len(np.unique(days)),
            "pinball_MW": d.loss.mean(), "vs_ref_%": 100 * diff.mean() / ref.loss.mean(),
            "ci95_%": f"[{100 * np.percentile(bd, 2.5) / ref.loss.mean():+.1f}, {100 * np.percentile(bd, 97.5) / ref.loss.mean():+.1f}]",
            "P(better)": float(np.mean(bd < 0)),
            "zone_skill_%": zs, "zone_skill_ci95": f"[{np.percentile(bz, 2.5):+.1f}, {np.percentile(bz, 97.5):+.1f}]",
            "system_total_skill_%": sys_d,
            "below_p10": d.below_p10.mean(), "below_p50": d.below_p50.mean(), "below_p90": d.below_p90.mean(),
            "cov_80": d.below_p90.mean() - d.below_p10.mean(), "width_MW": d.width.mean(),
            "within_1sd": float(np.mean(np.abs(zsig) < 1)), "within_2sd": float(np.mean(np.abs(zsig) < 2)),
            "beyond_3sd": float(np.mean(np.abs(zsig) > 3)),
            "parse_fail_%": 100 * (1 - d.parse_ok.mean()),
        })
    headline = pd.DataFrame(headline).sort_values("pinball_MW")

    slices = {}
    for col in ["models_agreed", "big_miss"] + [f"theme_{k}" for k in THEMES] + ["zone", "month"]:
        rows = []
        for name, d in per_ex.items():
            for val, g in d.groupby(col):
                rg = ref.loc[g.index]
                rows.append({"run": name, col: val, "n": len(g), "vs_ref_%": 100 * (g.loss.mean() / rg.loss.mean() - 1)})
        slices[col] = pd.DataFrame(rows).pivot(index="run", columns=col, values="vs_ref_%")
        counts = pd.DataFrame(rows).groupby(col)["n"].first()
        slices[col].columns = [f"{c} (n={counts[c]})" for c in slices[col].columns]

    zone_cal = {}
    for name, d in per_ex.items():
        zone_cal[name] = d.groupby("zone")[["below_p10", "below_p90"]].mean().apply(
            lambda r: f"{r.below_p10:.0%}/{r.below_p90:.0%}", axis=1)
    zone_cal = pd.DataFrame(zone_cal).T
    try:
        extras = _extra_metrics(long, meta, ref_name, ids, boots[:1000])
    except Exception as e:  # prices DB unavailable etc.: report without the extras
        extras = pd.DataFrame({"note": [f"extra metrics unavailable: {e}"]})
    hour_rows = []
    for name, d in long.items():
        for h, g in d.groupby("hour"):
            rg = long[ref_name][long[ref_name].hour == h]
            hour_rows.append({"run": name, "hour": f"HE{h}", "vs_ref_%": 100 * (g.loss.mean() / rg.loss.mean() - 1)})
    slices["hour_ending"] = pd.DataFrame(hour_rows).pivot(index="run", columns="hour", values="vs_ref_%")
    calib = pd.DataFrame([{"run": n, **{f"below_{k}": float(np.mean(d.y < d[k])) for k in Q}} for n, d in long.items()]).set_index("run")
    return {"split": split, "reference": ref_name, "headline": headline, "slices": slices, "zone_calibration": zone_cal,
            "extras": extras, "calibration": calib,
            "per_ex": per_ex, "examples": examples}


def dollars(rep: dict, train_split: str = "train") -> pd.DataFrame:
    """Commit-less-when-uncertain battery rule, per run, vs a fixed commitment (sim/settlement.py)."""
    from ercot_uncertainty.sim.settlement import backtest
    ref = {}
    for e in load_examples(train_split):
        ref.setdefault(e["zone"], []).append(np.mean([v["p90"] - v["p10"] for v in e["baseline"].values()]))
    ref = {z: float(np.median(v)) for z, v in ref.items()}
    rows = []
    for name, d in rep["per_ex"].items():
        w = pd.DataFrame({"zone": d.zone, "operating_date": d.day, "width": d.width})
        bt = backtest(w, ref)
        rows.append({"run": name, "days_simulated": len(bt), "fixed_rule_$": bt.fixed_usd.sum(),
                     "uncertainty_rule_$": bt.adaptive_usd.sum(), "uplift_$": bt.uplift_usd.sum(),
                     "uplift_%": 100 * bt.uplift_usd.sum() / abs(bt.fixed_usd.sum())})
    return pd.DataFrame(rows).sort_values("uplift_$", ascending=False)


def to_markdown(rep: dict, dollars_df: pd.DataFrame | None = None) -> str:
    fmt = lambda df: df.to_markdown(floatfmt=".3f")
    h = rep["headline"].copy()
    for c in ("below_p10", "below_p50", "below_p90", "cov_80", "within_1sd", "within_2sd", "beyond_3sd", "P(better)"):
        h[c] = (100 * h[c]).map(lambda x: f"{x:.1f}%")
    parts = [f"# Uncertainty eval report: `{rep['split']}`",
             f"Reference (declared in advance): **{rep['reference']}**. Negative `vs_ref_%` = lower pinball loss = better. "
             "`zone_skill_%` = mean over the 8 weather zones of each zone's % improvement vs the reference (equal weight; "
             "positive = better); the system total is reported on its own. CIs: day-clustered bootstrap. "
             "Unparseable answers are scored as p10=p50=p90=0 (\"trust ERCOT exactly\").",
             "## Headline", h.to_markdown(index=False, floatfmt=".2f"),
             "Calibration targets: below p10 = 10%, below p50 = 50%, below p90 = 90%; within 1 sigma = 68%, within 2 sigma = 95%, "
             "beyond 3 sigma = 0.3% (sigma view reads the p10-p90 range as +/-1.28 sigma).",
             "## Calibration by zone (share below p10 / below p90; targets 10% / 90%)", rep["zone_calibration"].to_markdown()]
    for k, t in rep["slices"].items():
        parts += [f"## Slice: {k} (% vs reference, negative = better)", fmt(t)]
    if "extras" in rep:
        parts += ["## Value-focused metrics",
                  "`price_weighted_pinball`: pinball loss weighted by the zone's day-ahead price that hour (skill where money is "
                  "at stake). `p50_vs_ERCOT_%`: mean absolute error of the predicted median vs ERCOT's own forecast (which "
                  "implies 0 error); negative = the median corrects ERCOT. `skill_vs_same_range_every_day_%`: pinball "
                  "improvement over the train-period p10/p50/p90 for that zone and hour.",
                  rep["extras"].to_markdown(index=False, floatfmt=".2f")]
    if dollars_df is not None:
        parts += ["## Dollars: commit less day-ahead when predicted uncertainty is high (1,000-unit fleet, price taker)",
                  dollars_df.to_markdown(index=False, floatfmt=",.0f")]
    return "\n\n".join(parts) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--split", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--dollars", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rep = build(a.split, [Path(f) for f in a.results], Path(a.reference))
    d = dollars(rep) if a.dollars else None
    md = to_markdown(rep, d)
    out = Path(a.out or f"results/report_{Path(a.split).stem}.md")
    out.write_text(md)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(rep["headline"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if d is not None:
            print(d.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
