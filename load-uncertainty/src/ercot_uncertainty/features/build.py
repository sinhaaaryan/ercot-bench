"""Cutoff-aware joins -> hourly feature frame -> numeric baselines -> prompt examples.

One example = (zone, operating day D), predicting the error of ERCOT's day-ahead in-use load
forecast (actual - forecast, MW) for PEAK_HOURS. Every input must be available strictly before
CUTOFF(D) = 10:00 America/Chicago on D-1; `assert_no_leakage` fails loudly otherwise.

Forecast sources
  ercot  data/parquet/lf_vintage.parquet (per zone, all ERCOT models, 'dam' + 'prev' vintages)
  eia    data/parquet/eia930_df.parquet  (FALLBACK: system total only; that forecast is ERCOT's
         14:30 D-1 post, i.e. after cutoff, so it is only the TARGET, never shown in the prompt)

  uv run python -m ercot_uncertainty.features.build --source ercot
  uv run python -m ercot_uncertainty.features.build --source eia
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from ercot_uncertainty.baselines.numeric import UPGRADED_PARAMS, gbm_oof_and_test, historical_quantiles
from ercot_uncertainty.baselines.scoring import QUANTILES, pinball
from ercot_uncertainty.data.ercot_time import cutoff_utc
from ercot_uncertainty.zones import ZONE_OFFICES

ROOT = Path(__file__).resolve().parents[3]
PQ = ROOT / "data" / "parquet"
OUT = ROOT / "data" / "examples"

PEAK_HOURS = [16, 17, 18, 19, 20]
KEY = ["zone", "operating_date", "hour_ending"]
AFD_MARGIN = pd.Timedelta(minutes=5)  # products stamped 09:59 may transmit a minute or two later
MAX_AFD_CHARS = 1800  # ~450 tokens of key messages / short term per office
MAX_LONG_TERM_CHARS = 1000  # plus the start of LONG TERM (covers the operating day when SHORT TERM is "today and tonight")
SYSTEM_OFFICES = ["FWD", "HGX", "EWX"]  # biggest load centres, for the system_total fallback
ERROR_LAG_DAYS = 2  # D-1 actuals are incomplete at the D-1 10:00 cutoff

# splits by operating day, with ~1 week embargo between them
SPLITS = {
    "train": (date(2023, 1, 1), date(2025, 8, 31)),
    "val": (date(2025, 9, 8), date(2026, 2, 28)),
    "test": (date(2026, 3, 8), date(2026, 8, 31)),
}

ZONE_LABEL = {
    "coast": "Coast (Houston)",
    "east": "East (Tyler/Lufkin)",
    "far_west": "Far West (Permian Basin)",
    "north": "North (Wichita Falls/Paris)",
    "north_central": "North Central (Dallas-Fort Worth)",
    "south": "Southern (Corpus Christi/Rio Grande Valley)",
    "south_central": "South Central (Austin/San Antonio)",
    "west": "West (Abilene/San Angelo)",
    "system_total": "ERCOT system total",
}


# --- forecasts + labels --------------------------------------------------------------------


def load_forecasts(source: str) -> pd.DataFrame:
    """Hourly frame: KEY + fc_inuse, fc_publish_utc, fc_visible, model-spread stats, revision."""
    if source == "eia":
        f = pd.read_parquet(PQ / "eia930_df.parquet")
        f = f[~f["dst_repeated_hour"]]
        out = f[["zone", "operating_date", "hour_ending"]].copy()
        out["fc_inuse"] = f["forecast_mw"].to_numpy()
        out["fc_publish_utc"] = f["inferred_publish_time_utc"].to_numpy()
        out["fc_visible"] = False
        return out

    f = pd.read_parquet(PQ / "lf_vintage.parquet")
    f = f[~f["dst_repeated_hour"]]
    dam, prev = f[f["vintage"] == "dam"], f[f["vintage"] == "prev"]
    wide = dam.pivot_table(index=KEY, columns="model", values="forecast_mw")
    stats = pd.DataFrame(
        {
            "n_models": wide.notna().sum(axis=1),
            "fc_min": wide.min(axis=1),
            "fc_max": wide.max(axis=1),
            "fc_std": wide.std(axis=1),
            "fc_median": wide.median(axis=1),
        }
    )
    inuse = dam[dam["in_use"]].groupby(KEY).agg(
        fc_inuse=("forecast_mw", "first"), inuse_model=("model", "first"), fc_publish_utc=("publish_time_utc", "max")
    )
    prev_inuse = prev[prev["in_use"]].groupby(KEY).agg(
        fc_prev=("forecast_mw", "first"), prev_publish_utc=("publish_time_utc", "max")
    )
    out = inuse.join(stats).join(prev_inuse).reset_index()
    # each ERCOT model's deviation from the in-use forecast (baseline audit: the strongest single input)
    dev = wide.sub(inuse["fc_inuse"].reindex(wide.index), axis=0)
    dev.columns = [f"dev_{m}" for m in dev.columns]
    out = out.join(dev, on=KEY)
    out["fc_range"] = out["fc_max"] - out["fc_min"]
    out["inuse_minus_median"] = out["fc_inuse"] - out["fc_median"]
    out["revision"] = out["fc_inuse"] - out["fc_prev"]
    out["fc_visible"] = True
    return out


def load_actuals() -> pd.DataFrame:
    a = pd.read_parquet(PQ / "actual_load.parquet")
    a = a[~a["dst_repeated_hour"]]
    return a[KEY + ["actual_mw"]]


# --- context features ----------------------------------------------------------------------


def add_recent_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Same-zone, same-hour errors from days <= D - ERROR_LAG_DAYS (strictly causal)."""
    df = df.sort_values(KEY).copy()
    d = pd.to_datetime(df["operating_date"])
    df["_d"] = d
    parts = []
    for (_, _), g in df.groupby(["zone", "hour_ending"], sort=False):
        s = g.set_index("_d")["error"].asfreq("D")  # calendar-regular so shifts are in days
        lagged = s.shift(ERROR_LAG_DAYS)
        feats = pd.DataFrame(
            {
                "err_lag2": lagged,
                "err_mean7": lagged.rolling(7, min_periods=4).mean(),
                "err_mae7": lagged.abs().rolling(7, min_periods=4).mean(),
                "err_mae30": lagged.abs().rolling(30, min_periods=15).mean(),
            }
        )
        parts.append(g.join(feats, on="_d"))
    return pd.concat(parts).drop(columns="_d")


def add_4cp_context(df: pd.DataFrame) -> pd.DataFrame:
    """Summer 4CP: big commercial loads curtail on likely system-peak days, so load tends to come in under
    forecast. Feature = ERCOT's system forecast peak for D / highest actual system hourly load so far this
    month (days <= D-2; falls back to the previous month early in the month)."""
    sys = df[df["zone"] == "system_total"]
    if sys.empty or not df["fc_visible"].all():
        df["sys_peak_ratio"] = np.nan
        return df
    fc_peak = sys.groupby("operating_date")["fc_inuse"].max()
    act = sys.groupby("operating_date")["actual_mw"].max()
    act.index = pd.to_datetime(act.index)
    ratios = {}
    for d, peak in fc_peak.items():
        dt = pd.Timestamp(d)
        hist = act[(act.index <= dt - pd.Timedelta(days=ERROR_LAG_DAYS)) & (act.index >= dt.replace(day=1) - pd.Timedelta(days=31))]
        this_month = hist[hist.index.month == dt.month]
        ref = this_month.max() if len(this_month) >= 3 else hist.max()
        ratios[d] = peak / ref if pd.notna(ref) and ref > 0 else np.nan
    df["sys_peak_ratio"] = df["operating_date"].map(ratios)
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["operating_date"])
    hol = USFederalHolidayCalendar().holidays(d.min(), d.max())
    df["month"] = d.dt.month
    df["dow"] = d.dt.dayofweek
    df["is_holiday"] = d.isin(hol)
    local_day = d.dt.tz_localize("America/Chicago")
    n_hours = ((local_day + pd.Timedelta(days=1)).dt.tz_convert("UTC") - local_day.dt.tz_convert("UTC"))
    df["dst_change_day"] = n_hours.dt.total_seconds().ne(86400).to_numpy()
    return df


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    path = PQ / "weather_d2.parquet"
    if not path.exists():
        print("weather_d2.parquet missing: weather features skipped")
        return df
    from ercot_uncertainty.data.weather import zone_hourly

    w = pd.read_parquet(path)
    w = zone_hourly(w[~w["dst_repeated_hour"]]).drop(columns=["dst_repeated_hour"])
    temp_models = [c for c in w.columns if c.startswith("temp_c__")]
    cloud_models = [c for c in w.columns if c.startswith("cloud_cover_pct__")]
    w["temp_spread_c"] = w[temp_models].std(axis=1) if len(temp_models) > 1 else np.nan
    w["cloud_spread_pct"] = w[cloud_models].std(axis=1) if len(cloud_models) > 1 else np.nan
    # day-over-day change of the forecast temperature (the D-1 value is an even older forecast)
    w = w.sort_values(["zone", "hour_ending", "operating_date"])
    w["temp_delta_d1"] = w.groupby(["zone", "hour_ending"])["temp_c"].diff()
    keep = ["temp_c", "dewpoint_c", "cloud_cover_pct", "wind_speed_kmh", "precip_mm", "temp_spread_c",
            "cloud_spread_pct", "temp_delta_d1", "issued_utc_max"]
    w = w[["zone", "operating_date", "hour_ending"] + [c for c in keep if c in w]]
    w = w.rename(columns={"issued_utc_max": "wx_issued_utc"})

    if (df["zone"] == "system_total").any():  # load-weighted zone average for the system fallback
        a = load_actuals()
        weights = a[a["zone"] != "system_total"].groupby("zone")["actual_mw"].mean()
        weights /= weights.sum()
        ww = w.assign(_w=w["zone"].map(weights))
        num_cols = [c for c in keep if c in w and c != "issued_utc_max"]
        agg = ww.groupby(["operating_date", "hour_ending"]).apply(
            lambda g: pd.Series({c: np.average(g[c].fillna(g[c].mean()), weights=g["_w"]) for c in num_cols}
                                | {"wx_issued_utc": g["wx_issued_utc"].max()}),
            include_groups=False,
        ).reset_index()
        w = pd.concat([w, agg.assign(zone="system_total")], ignore_index=True)
    return df.merge(w, on=KEY, how="left")


# --- forecaster text -----------------------------------------------------------------------

_DATE_LINE = re.compile(r"^\s*\d{3,4}\s+[AP]M\s+[A-Z]{3,4}\s+\w{3}\s+\w{3}\s+\d{1,2}\s+\d{4}\s*$", re.M)
_ISSUED = re.compile(r"/\s*Issued[^/]*/|^\s*(Issued|Updated) at .*$", re.I | re.M)
_STAMP = re.compile(r"\b\d{3,4}\s+[AP]M\s+[A-Z]{3,4}\s+(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+[A-Z][a-z]{2}\s+\d{1,2}(\s+\d{4})?")
_YEAR = re.compile(r"\b20[12]\d\b")


def clean_afd(text: str, max_chars: int = MAX_AFD_CHARS) -> str:
    """Strip issue-time lines and years (limits date memorization), collapse whitespace, cap length."""
    text = _STAMP.sub("", _ISSUED.sub("", _DATE_LINE.sub("", text or "")))
    text = _YEAR.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " [...]"
    return text


def long_term_head(sections_json: str | None) -> str:
    if not isinstance(sections_json, str):
        return ""
    secs = json.loads(sections_json)
    key = next((k for k in secs if "LONG" in k.upper()), None)
    return secs[key] if key else ""


def afd_prompt_text(reasoning_text: str, sections_json: str | None) -> str:
    text = clean_afd(reasoning_text)
    lt = long_term_head(sections_json)
    if lt and lt.strip()[:200] not in reasoning_text:
        text += "\n\n.LONG TERM...\n" + clean_afd(lt, MAX_LONG_TERM_CHARS)
    return text


def afd_for_days(zone_days: pd.DataFrame, afd: pd.DataFrame | None = None) -> pd.DataFrame:
    """Latest AFD per (zone, operating_date) from its covering office(s), issued < cutoff - margin."""
    if afd is None:
        afd = pd.read_parquet(PQ / "afd.parquet", columns=["office", "issued_utc", "reasoning_text", "sections_json"])
    if "sections_json" not in afd:
        afd = afd.assign(sections_json=None)
    afd = afd[afd["reasoning_text"].str.strip().str.len() > 0].sort_values("issued_utc")
    rows = []
    for zone, g in zone_days.groupby("zone"):
        offices = SYSTEM_OFFICES if zone == "system_total" else ZONE_OFFICES[zone][:1]
        t = g[["operating_date"]].drop_duplicates().copy()
        t["cutoff_utc"] = t["operating_date"].map(cutoff_utc)
        t["lookup"] = t["cutoff_utc"] - AFD_MARGIN - pd.Timedelta(microseconds=1)  # strictly before
        t = t.sort_values("lookup")
        for office in offices:
            a = afd[afd["office"] == office][["issued_utc", "reasoning_text", "sections_json"]]
            m = pd.merge_asof(t, a, left_on="lookup", right_on="issued_utc", direction="backward")
            m = m.dropna(subset=["issued_utc"])
            m["zone"], m["office"] = zone, office
            rows.append(m[["zone", "operating_date", "office", "issued_utc", "cutoff_utc", "reasoning_text",
                           "sections_json"]])
    return pd.concat(rows, ignore_index=True)


# --- leakage guard -------------------------------------------------------------------------


def assert_no_leakage(hourly: pd.DataFrame, afd: pd.DataFrame) -> None:
    cut = hourly["operating_date"].map(cutoff_utc)
    checks = {"afd": (afd["issued_utc"], afd["cutoff_utc"] - AFD_MARGIN)}
    if hourly["fc_visible"].all():
        checks["forecast_dam"] = (hourly["fc_publish_utc"], cut)
        if "prev_publish_utc" in hourly:
            checks["forecast_prev"] = (hourly["prev_publish_utc"], cut)
    if "wx_issued_utc" in hourly:
        checks["weather"] = (hourly["wx_issued_utc"], cut)
    for name, (t, c) in checks.items():
        t = pd.to_datetime(t, utc=True)
        bad = t.notna() & ~(t < c)
        if bad.any():
            raise AssertionError(f"LEAKAGE in {name}: {int(bad.sum())} inputs at/after cutoff, e.g.\n"
                                 f"{hourly.loc[bad[bad].index[:3]] if name != 'afd' else afd[bad].head(3)}")
    print("leakage check passed:", ", ".join(checks))


# --- prompts -------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an ERCOT load-forecast risk analyst. For one ERCOT weather zone and one operating day, predict the distribution of the ERROR of ERCOT's day-ahead load forecast (actual load minus ERCOT's forecast, in MW) for each listed hour-ending. Positive error = load came in above ERCOT's forecast.

Everything you are given was available at 10:00 Central on the day before the operating day. A statistical model's p10/p50/p90 of the error is included; it uses the numbers but cannot read the forecaster discussion. Use it as your anchor and widen, tighten, or shift it when the evidence says the weather forecast itself is unusually uncertain (fronts, convection, cloud cover over solar, timing) or unusually settled.

You are scored with pinball loss on p10/p50/p90 against the realized error: intervals that are too wide are penalized just like intervals that are too narrow.

Respond with JSON only, no prose, no code fences:
{"hours":[{"he":16,"p10":-900,"p50":50,"p90":1100}, ...]}
one entry per listed hour-ending, integer MW, p10 <= p50 <= p90."""

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
         "October", "November", "December"]


def _f(c: float | None) -> str:
    return "n/a" if c is None or pd.isna(c) else f"{c * 9 / 5 + 32:.0f}"


def _mw(x: float | None, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+,.0f}" if signed else f"{x:,.0f}"


def build_user_prompt(zone: str, day: pd.DataFrame, afd_rows: pd.DataFrame, variant: str) -> str:
    r0 = day.iloc[0]
    lines = [
        f"Zone: {ZONE_LABEL[zone]}",
        f"Operating day: a {DOW[r0['dow']]} in {MONTH[r0['month'] - 1]}"
        + (" (US federal holiday)" if r0["is_holiday"] else "")
        + (" (daylight-saving change day)" if r0["dst_change_day"] else ""),
        "",
    ]
    show_numeric = variant in ("full", "numeric")
    if show_numeric:
        head = ["HE"]
        if r0["fc_visible"]:
            head += ["ERCOT fcst MW", f"model range ({int(r0['n_models'])} models)", "in-use vs median",
                     "revision since prior day"]
        has_wx = "temp_c" in day and day["temp_c"].notna().any()
        if has_wx:
            head += ["temp F", "dewpt F", "cloud %", "temp spread F (wx models)"]
        head += ["stat p10 / p50 / p90"]
        lines.append(" | ".join(head))
        for _, r in day.iterrows():
            row = [str(r["hour_ending"])]
            if r0["fc_visible"]:
                row += [_mw(r["fc_inuse"]), f"{_mw(r['fc_min'])} .. {_mw(r['fc_max'])}",
                        _mw(r["inuse_minus_median"], True), _mw(r["revision"], True)]
            if has_wx:
                spread = "n/a" if pd.isna(r["temp_spread_c"]) else f"{r['temp_spread_c'] * 9 / 5:.1f}"
                row += [_f(r["temp_c"]), _f(r["dewpoint_c"]),
                        "n/a" if pd.isna(r["cloud_cover_pct"]) else f"{r['cloud_cover_pct']:.0f}", spread]
            row += [f"{_mw(r['p10'], True)} / {_mw(r['p50'], True)} / {_mw(r['p90'], True)}"]
            lines.append(" | ".join(row))
        if r0["fc_visible"]:
            lines.append(f"ERCOT's in-use model for this day: {r0['inuse_model']}")
        if r0.get("is_4cp_season") and pd.notna(r0.get("sys_peak_ratio")):
            lines.append(f"Summer 4CP season: ERCOT's system-wide forecast peak for this day is "
                         f"{100 * r0['sys_peak_ratio']:.0f}% of the highest actual system load in the last few weeks "
                         f"(large commercial loads curtail on likely peak days).")
        e = day[["err_mean7", "err_mae7", "err_mae30"]].mean()
        lines += [
            "",
            "Recent ERCOT day-ahead errors for this zone at these hours (actual - forecast), through two days ago:",
            f"  most recent available day: {', '.join(_mw(x, True) for x in day['err_lag2'])}"
            f"; last 7 days mean {_mw(e['err_mean7'], True)}, mean absolute {_mw(e['err_mae7'])}"
            f"; last 30 days mean absolute {_mw(e['err_mae30'])}",
        ]
    else:
        lines.append("HE | stat p10 / p50 / p90")
        for _, r in day.iterrows():
            lines.append(f"{r['hour_ending']} | {_mw(r['p10'], True)} / {_mw(r['p50'], True)} / {_mw(r['p90'], True)}")

    if variant in ("full", "text"):
        for _, a in afd_rows.iterrows():
            if not isinstance(a["reasoning_text"], str):
                continue
            age_h = (a["cutoff_utc"] - a["issued_utc"]).total_seconds() / 3600
            lines += ["", f"Latest NWS {a['office']} Area Forecast Discussion (issued {age_h:.0f} h before the cutoff):",
                      '"""', afd_prompt_text(a["reasoning_text"], a.get("sections_json")), '"""']
    lines += ["", f"Predict the error distribution for hour-endings {', '.join(map(str, day['hour_ending']))}."]
    return "\n".join(lines)


# --- main pipeline -------------------------------------------------------------------------


def split_of(d: date) -> str | None:
    for name, (lo, hi) in SPLITS.items():
        if lo <= d <= hi:
            return name
    return None


def build_hourly(source: str) -> pd.DataFrame:
    fc = load_forecasts(source)
    df = fc.merge(load_actuals(), on=KEY, how="inner")
    df["error"] = df["actual_mw"] - df["fc_inuse"]
    df = add_recent_errors(df)  # uses all hours so lags see every day
    # NOTE: no D-1 errors at all. ERCOT posts zonal actual load once a day (05:50 CT, for the previous day),
    # so the latest actuals available at the D-1 10:00 cutoff are for D-2 (see docs/REVIEW.md B1).
    df = add_4cp_context(df)
    df = df[df["hour_ending"].isin(PEAK_HOURS)].copy()
    df = add_calendar(df)
    df["is_4cp_season"] = df["month"].between(6, 9) & (df["dow"] < 5) & ~df["is_holiday"]
    df = add_weather(df)
    if source == "eia":
        for c in ["n_models", "fc_min", "fc_max", "fc_std", "fc_range", "inuse_minus_median", "revision"]:
            df[c] = np.nan
        df["inuse_model"] = None
    df["split"] = df["operating_date"].map(split_of)
    return df[df["split"].notna()].reset_index(drop=True)


FEATURES = ["zone", "hour_ending", "month", "dow", "is_holiday", "fc_inuse", "fc_std", "fc_range",
            "inuse_minus_median", "revision", "err_lag2", "err_mean7", "err_mae7", "err_mae30", "temp_c",
            "dewpoint_c", "cloud_cover_pct", "wind_speed_kmh", "precip_mm", "temp_spread_c",
            "cloud_spread_pct", "temp_delta_d1", "sys_peak_ratio", "is_4cp_season"]


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in FEATURES if c in df and df[c].notna().any()]
    if not df["fc_visible"].all():
        feats = [c for c in feats if c != "fc_inuse"]  # post-cutoff in the EIA fallback
    for c in ("is_holiday", "is_4cp_season"):
        if c in df:
            df[c] = df[c].astype(int)
    hist = historical_quantiles(df[KEY + ["error"]], window_days=90).rename(
        columns={k: f"hist_{k}" for k in QUANTILES})
    df = df.merge(hist, on=KEY, how="left")
    feats += [f"hist_{k}" for k in QUANTILES]
    train = df[df["split"] == "train"]
    rest = df[df["split"] != "train"]
    # v1 = the original config (kept as a leaderboard reference row)
    oof, pred = gbm_oof_and_test(train, rest, feats)
    v1 = pd.concat([oof, pred], ignore_index=True).rename(columns={k: f"gbm_v1_{k}" for k in QUANTILES})
    df = df.merge(v1, on=KEY, how="left")
    # v2 = audited upgrade: + per-model deviations + in-use model identity, bigger trees, per-zone calibration
    dev_cols = sorted(c for c in df.columns if c.startswith("dev_"))
    feats_v2 = feats + dev_cols + (["inuse_model"] if df["inuse_model"].notna().any() else [])
    oof, pred = gbm_oof_and_test(train, rest, feats_v2, params=UPGRADED_PARAMS,
                                 categorical=["zone", "inuse_model"], calibrate=True)
    gbm = pd.concat([oof, pred], ignore_index=True).rename(columns={k: f"gbm_{k}" for k in QUANTILES})
    df = df.merge(gbm, on=KEY, how="left")
    print("GBM v2 features:", feats_v2)

    # reward reference = whichever numeric baseline has lower pinball loss on val
    val = df[(df["split"] == "val") & df["error"].notna()
             & df[[f"{n}_{k}" for n in ("hist", "gbm") for k in QUANTILES]].notna().all(axis=1)]
    loss = {name: np.mean([pinball(val["error"], val[f"{name}_{k}"], tau).mean() for k, tau in QUANTILES.items()])
            for name in ("gbm", "hist")}
    loss_v1 = np.mean([pinball(val["error"], val[f"gbm_v1_{k}"], tau).mean() for k, tau in QUANTILES.items()])
    print(f"val pinball MW: gbm_v1 (original) {loss_v1:.2f}")
    best = min(loss, key=loss.get)
    print(f"val pinball MW: {loss} -> reward baseline = {best}")
    for k in QUANTILES:
        df[k] = df[f"{best}_{k}"].fillna(df[f"gbm_{k}"])
    df.attrs["baseline_name"] = best
    return df


def write_examples(df: pd.DataFrame, afd: pd.DataFrame, source: str) -> dict:
    agree_thr = df[df["split"] == "train"].groupby("zone")["fc_range"].median()
    miss_thr = df[df["split"] == "train"].groupby("zone")["error"].apply(lambda s: s.abs().quantile(0.8))
    afd_by = {k: g for k, g in afd.groupby(["zone", "operating_date"])}
    counts: dict[str, int] = {}
    files = {}
    for variant in ("full", "numeric", "text"):
        d = OUT if variant == "full" else OUT / variant
        d.mkdir(parents=True, exist_ok=True)
        files[variant] = {s: (d / f"{s}.jsonl").open("w") for s in SPLITS}

    for (zone, od), day in df.groupby(["zone", "operating_date"]):
        day = day.sort_values("hour_ending")
        if list(day["hour_ending"]) != PEAK_HOURS or day[["error", "p10", "p50", "p90"]].isna().any().any():
            continue
        split = day["split"].iloc[0]
        baseline = {str(r.hour_ending): {k: round(float(getattr(r, k)), 1) for k in QUANTILES}
                    for r in day.itertuples()}
        mean_range = day["fc_range"].mean()
        meta = {
            "source": source,
            "models_agreed": bool(mean_range <= agree_thr.get(zone, np.nan)) if pd.notna(mean_range) else None,
            "big_miss": bool(day["error"].abs().max() >= miss_thr[zone]),
            "fc_range_mean": None if pd.isna(mean_range) else round(float(mean_range), 1),
            "inuse_model": day["inuse_model"].iloc[0],
            "baseline_name": df.attrs.get("baseline_name"),
            "gbm_baseline": {str(r.hour_ending): {k: round(float(getattr(r, f"gbm_{k}")), 1) for k in QUANTILES}
                             for r in day.itertuples()},
            "gbm_v1_baseline": {str(r.hour_ending): {k: round(float(getattr(r, f"gbm_v1_{k}")), 1) for k in QUANTILES}
                                for r in day.itertuples()},
            "hist_baseline": {str(r.hour_ending): {k: None if pd.isna(getattr(r, f"hist_{k}")) else
                                                   round(float(getattr(r, f"hist_{k}")), 1) for k in QUANTILES}
                              for r in day.itertuples()},
            "afd": [{"office": a.office, "issued_utc": str(a.issued_utc)}
                    for a in afd_by.get((zone, od), pd.DataFrame()).itertuples()],
            "cutoff_utc": str(cutoff_utc(od)),
        }
        base = {
            "id": f"{od}_{zone}", "zone": zone, "operating_date": str(od), "split": split,
            "hours": PEAK_HOURS, "baseline": baseline,
            "actual_error": {str(r.hour_ending): round(float(r.error), 1) for r in day.itertuples()},
            "meta": meta,
        }
        afd_rows = afd_by.get((zone, od), pd.DataFrame(columns=["office", "issued_utc", "cutoff_utc", "reasoning_text"]))
        for variant, fh in files.items():
            prompt = [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": build_user_prompt(zone, day, afd_rows, variant)}]
            fh[split].write(json.dumps({**base, "prompt": prompt}) + "\n")
        counts[split] = counts.get(split, 0) + 1
    for fh in files.values():
        for f in fh.values():
            f.close()
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["ercot", "eia"], default="ercot")
    a = p.parse_args()
    df = build_hourly(a.source)
    afd = afd_for_days(df[["zone", "operating_date"]])
    assert_no_leakage(df, afd)
    df = add_baselines(df)
    df.to_parquet(PQ / f"hourly_features_{a.source}.parquet")
    counts = write_examples(df, afd, a.source)
    print("examples per split:", counts)


if __name__ == "__main__":
    main()
