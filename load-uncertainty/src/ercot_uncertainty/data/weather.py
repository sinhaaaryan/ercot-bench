"""Day-ahead weather forecasts (Open-Meteo Previous Runs API, ``*_previous_day2``) per ERCOT
weather-zone point, strictly issued before CUTOFF(D) = 10:00 America/Chicago on D-1.

Source: ``https://previous-runs-api.open-meteo.com/v1/forecast``. Raw JSON responses are cached
under ``data/raw/weather/<model>/``; output is ``data/parquet/weather_d2.parquet``.

Lead / issue-time semantics (verified 2026-09-26, see ``verify_semantics`` and DATA_CONTRACT):

* Open-Meteo writes every model run into the ``previous_dayN`` series for all timesteps with
  lead >= N*24h, and later runs overwrite earlier ones (open-meteo ``OmFileSplitter.
  updateFromTimeOrientedStreaming3D``: ``skip = previousDay * 86400 / dt``). So the
  ``previous_day2`` value at valid time t comes from the **latest run initialised <= t - 48h**.
  Checked against the Single Runs API: gfs_global and icon_global 48/48 hours equal the run
  ``floor_6h(t - 48h)``; gem_global equals ``floor_12h(t - 48h)`` (00/12Z runs only);
  ecmwf_ifs025 matches on its native 3-hourly grid, while off-grid hours are Hermite-interpolated at
  read time from stored 3-hourly values, so they may also depend on the run ``floor_6h(t+2..5h - 48h)``.
* Worst case for operating day D is HE24 (valid 00:00 local on D+1 = 05Z/06Z): the newest run that
  can contribute is 06Z on D-1 (00Z for GEM). Open-Meteo's own run availability is ~3.6h (ICON),
  ~5.5-7.4h (GFS), ~7.1h (ECMWF), ~6h (GEM) after init, i.e. <= ~13:30Z on D-1, before the
  cutoff (15Z CDT / 16Z CST). ``issued_utc_max`` stores a conservative bound per row
  (latest contributing run init + ``latency_h``) and ``assert_issued_before_cutoff`` checks it.
* ``previous_day1`` would be pre-cutoff only for roughly HE1-HE5 (run init <= 06Z D-1), so it is
  not used.

Hour mapping: Open-Meteo hourly timestamps are instants (temperature, dew point, cloud, wind) or
the sum over the preceding hour (precipitation). ``valid_time_utc`` = the timestamp; the ERCOT
interval is the hour ending at it (interval start = valid - 1h), mapped with
``ercot_time.he_from_interval_start_utc`` (DST days have 23/25 hours).

CLI::

    uv run python -m ercot_uncertainty.data.weather --start 2023-01-01 --end 2026-08-31
    uv run python -m ercot_uncertainty.data.weather --verify   # re-check day2 semantics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ercot_uncertainty.data.ercot_time import cutoff_utc, he_from_interval_start_utc
from ercot_uncertainty.zones import POINTS, POINTS_BY_NAME, Point

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
PARQUET_PATH = PROJECT_ROOT / "data" / "parquet" / "weather_d2.parquet"

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
LEAD_DAYS = 2

# Open-Meteo variable -> contract column
VARIABLES = {
    "temperature_2m": "temp_c",
    "dew_point_2m": "dewpoint_c",
    "cloud_cover": "cloud_cover_pct",
    "wind_speed_10m": "wind_speed_kmh",
    "precipitation": "precip_mm",
}
VALUE_COLS = list(VARIABLES.values())


@dataclass(frozen=True)
class ModelSpec:
    key: str  # column suffix
    om_model: str  # Open-Meteo `models=` value
    cycle_h: int  # hours between runs
    step_h: int  # native output step near 48h lead
    latency_h: float  # conservative bound: run init -> available on Open-Meteo
    first_date: date  # earliest valid date with previous_day2 data (any variable)


MODELS: dict[str, ModelSpec] = {
    # GFS temperature_2m goes back to 2021-03-25; its other variables start 2024-01-20.
    "gfs": ModelSpec("gfs", "gfs_global", 6, 1, 8.0, date(2021, 3, 25)),
    "ecmwf": ModelSpec("ecmwf", "ecmwf_ifs025", 6, 3, 8.0, date(2024, 2, 4)),
    "icon": ModelSpec("icon", "icon_global", 6, 1, 5.0, date(2024, 1, 20)),
    "gem": ModelSpec("gem", "gem_global", 12, 3, 7.0, date(2024, 1, 20)),
}

# Polite pacing. Open-Meteo free tier: 600 calls/min, 5000/hour, 10000/day, where one request
# counts as n_locations * max(1, n_vars/10) * max(1, n_days/14) calls.
CHUNK_DAYS = 183
MINUTE_BUDGET = 500.0
HOUR_BUDGET = 4500.0


# ----------------------------------------------------------------------------- issue-time bound
def latest_run_init_utc(valid_utc: pd.Series, spec: ModelSpec, lead_days: int = LEAD_DAYS) -> pd.Series:
    """Newest run init that can influence the ``previous_day{lead_days}`` value at ``valid_utc``.

    Stored series holds, per native step, the latest run with lead >= lead_days*24h. For
    sub-native (interpolated) hours, Hermite interpolation reads up to 2 native steps ahead, so we
    use ceil_to_step(t) + step as the effective valid time (conservative)."""
    t = pd.to_datetime(valid_utc, utc=True)
    if spec.step_h > 1:
        on_grid = (t.dt.hour % spec.step_h == 0) & (t.dt.minute == 0)
        t_eff = t.where(on_grid, t.dt.ceil(f"{spec.step_h}h") + pd.Timedelta(hours=spec.step_h))
    else:
        t_eff = t
    return (t_eff - pd.Timedelta(days=lead_days)).dt.floor(f"{spec.cycle_h}h")


def issued_utc_bound(valid_utc: pd.Series, spec: ModelSpec, lead_days: int = LEAD_DAYS) -> pd.Series:
    return latest_run_init_utc(valid_utc, spec, lead_days) + pd.Timedelta(hours=spec.latency_h)


def assert_issued_before_cutoff(
    df: pd.DataFrame, time_col: str = "issued_utc_max", date_col: str = "operating_date"
) -> None:
    """Raise AssertionError unless every ``time_col`` is strictly before CUTOFF(operating_date).

    NaT counts as a violation (unknown issue time)."""
    if df.empty:
        return
    dates = pd.Series(pd.unique(df[date_col]))
    cut = pd.Series([cutoff_utc(d) for d in dates], index=dates.values)
    cutoffs = pd.to_datetime(df[date_col].map(cut), utc=True)
    issued = pd.to_datetime(df[time_col], utc=True)
    bad = ~(issued < cutoffs)  # NaT -> False -> bad
    if bad.any():
        sample = df.loc[bad, [date_col, time_col]].head(5).assign(cutoff_utc=cutoffs[bad].head(5))
        raise AssertionError(
            f"{int(bad.sum())} rows have {time_col} >= CUTOFF(operating_date) or NaT:\n{sample}"
        )


# ----------------------------------------------------------------------------- fetching
class _Pacer:
    """Sliding-window budget on Open-Meteo weighted call counts."""

    def __init__(self) -> None:
        self.events: deque[tuple[float, float]] = deque()

    def wait(self, weight: float) -> None:
        while True:
            now = time.time()
            while self.events and now - self.events[0][0] > 3600:
                self.events.popleft()
            last_min = sum(w for t, w in self.events if now - t <= 60)
            last_hour = sum(w for _, w in self.events)
            if last_min + weight <= MINUTE_BUDGET and last_hour + weight <= HOUR_BUDGET:
                self.events.append((now, weight))
                return
            time.sleep(5)


_PACER = _Pacer()


def _call_weight(n_loc: int, n_vars: int, start: date, end: date) -> float:
    n_days = (end - start).days + 1
    return n_loc * max(1.0, n_vars / 10) * max(1.0, n_days / 14)


def _cache_path(spec: ModelSpec, points: list[Point], start: date, end: date) -> Path:
    key = json.dumps(
        [spec.om_model, [(p.name, p.lat, p.lon) for p in points], sorted(VARIABLES), LEAD_DAYS]
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    return RAW_DIR / spec.om_model / f"{start:%Y%m%d}_{end:%Y%m%d}_{h}.json"


def fetch_chunk(spec: ModelSpec, points: list[Point], start: date, end: date, refresh: bool = False) -> list[dict]:
    """One request for all ``points`` over [start, end] (UTC dates). Cached as raw JSON."""
    path = _cache_path(spec, points, start, end)
    if path.exists() and not refresh:
        return json.loads(path.read_text())["responses"]
    hourly = ",".join(f"{v}_previous_day{LEAD_DAYS}" for v in VARIABLES)
    params = {
        "latitude": ",".join(f"{p.lat}" for p in points),
        "longitude": ",".join(f"{p.lon}" for p in points),
        "hourly": hourly,
        "models": spec.om_model,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "GMT",
    }
    weight = _call_weight(len(points), len(VARIABLES), start, end)
    for attempt in range(8):
        _PACER.wait(weight)
        try:
            r = requests.get(API_URL, params=params, timeout=180)
        except requests.RequestException as e:
            print(f"  network error {e!r}; retry", file=sys.stderr)
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code == 429 or (r.status_code == 400 and "limit" in r.text.lower()):
            msg = r.text.lower()
            if "daily" in msg:
                raise RuntimeError(f"Open-Meteo daily limit hit: {r.text[:200]}")
            wait = 900 if "hourly" in msg else 65
            print(f"  rate limited ({r.text[:120]}); sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        responses = data if isinstance(data, list) else [data]
        if len(responses) != len(points):
            raise RuntimeError(f"expected {len(points)} locations, got {len(responses)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"params": params, "points": [p.name for p in points],
                                   "responses": responses}))
        tmp.replace(path)
        return responses
    raise RuntimeError(f"failed to fetch {spec.om_model} {start}..{end}")


def _chunks(start: date, end: date, days: int = CHUNK_DAYS) -> list[tuple[date, date]]:
    out, s = [], start
    while s <= end:
        e = min(end, s + timedelta(days=days - 1))
        out.append((s, e))
        s = e + timedelta(days=1)
    return out


def _responses_to_frame(responses: list[dict], points: list[Point], suffix: str) -> pd.DataFrame:
    frames = []
    for p, resp in zip(points, responses):
        h = resp["hourly"]
        df = pd.DataFrame({"valid_time_utc": pd.to_datetime(h["time"], utc=True)})
        for v, col in VARIABLES.items():
            df[f"{col}__{suffix}"] = pd.to_numeric(
                pd.Series(h.get(f"{v}_previous_day{LEAD_DAYS}", [None] * len(df))), errors="coerce"
            ).astype("float32")
        df["point"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pull_model(spec: ModelSpec, start: date, end: date, points: list[Point] = POINTS,
               refresh: bool = False) -> pd.DataFrame:
    """UTC-date range [start, end] for one model, all points."""
    start = max(start, spec.first_date)
    if start > end:
        return pd.DataFrame()
    frames = []
    for s, e in _chunks(start, end):
        print(f"{spec.om_model}: {s}..{e}", file=sys.stderr, flush=True)
        frames.append(_responses_to_frame(fetch_chunk(spec, points, s, e, refresh), points, spec.key))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["point", "valid_time_utc"], keep="last")


# ----------------------------------------------------------------------------- build
def build(start: date, end: date, models: list[str] | None = None, refresh: bool = False) -> pd.DataFrame:
    """Operating dates [start, end] -> point-level hourly frame (see DATA_CONTRACT)."""
    models = models or list(MODELS)
    # Operating day D spans valid times (D 01:00 local, D+1 00:00 local] = UTC dates D .. D+1.
    utc_start, utc_end = start, end + timedelta(days=1)
    merged: pd.DataFrame | None = None
    for m in models:
        df = pull_model(MODELS[m], utc_start, utc_end, refresh=refresh)
        if df.empty:
            continue
        merged = df if merged is None else merged.merge(df, on=["point", "valid_time_utc"], how="outer")
    assert merged is not None, "no data"
    return assemble(merged, [m for m in models if any(c.endswith(f"__{m}") for c in merged)], start, end)


def assemble(merged: pd.DataFrame, models: list[str], start: date, end: date) -> pd.DataFrame:
    """Per-model point frame (point, valid_time_utc, <col>__<model>...) -> contract frame."""
    merged = merged.copy()
    # model-mean base columns and model count
    for col in VALUE_COLS:
        cols = [f"{col}__{m}" for m in models if f"{col}__{m}" in merged]
        for c in cols:
            merged[c] = merged[c].astype("float32")
        merged[col] = merged[cols].mean(axis=1, skipna=True).astype("float32")
    temp_cols = [f"temp_c__{m}" for m in models if f"temp_c__{m}" in merged]
    merged["n_models"] = merged[temp_cols].notna().sum(axis=1).astype("int8")
    has_any = merged[[f"{c}__{m}" for c in VALUE_COLS for m in models if f"{c}__{m}" in merged]].notna().any(axis=1)
    merged = merged[has_any].copy()

    # conservative issue-time bound: newest contributing run across models with data in the row
    bounds = []
    for m in models:
        spec = MODELS[m]
        present = merged[[c for c in merged if c.endswith(f"__{m}")]].notna().any(axis=1)
        b = issued_utc_bound(merged["valid_time_utc"], spec).where(present)
        bounds.append(b)
    merged["issued_utc_max"] = pd.concat(bounds, axis=1).max(axis=1)
    merged["issued_utc_max"] = pd.to_datetime(merged["issued_utc_max"], utc=True)

    # ERCOT operating day / hour ending (interval ending at valid time)
    he = he_from_interval_start_utc(merged["valid_time_utc"] - pd.Timedelta(hours=1))
    merged = pd.concat([merged, he], axis=1)
    merged["zone"] = merged["point"].map(lambda n: POINTS_BY_NAME[n].zone)
    merged = merged[(merged["operating_date"] >= start) & (merged["operating_date"] <= end)]

    front = ["operating_date", "hour_ending", "dst_repeated_hour", "zone", "point", "valid_time_utc",
             *VALUE_COLS, "n_models", "issued_utc_max"]
    per_model = [f"{c}__{m}" for c in VALUE_COLS for m in models if f"{c}__{m}" in merged]
    out = merged[front + per_model].sort_values(["zone", "point", "valid_time_utc"]).reset_index(drop=True)
    assert_issued_before_cutoff(out)
    return out


def zone_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Load-weighted average of point rows to one row per (operating_date, hour_ending,
    dst_repeated_hour, zone). NaNs are skipped (weights renormalised)."""
    val = [c for c in df.columns if c.split("__")[0] in VALUE_COLS]
    w = df["point"].map(lambda n: POINTS_BY_NAME[n].weight).astype(float)
    keys = ["operating_date", "hour_ending", "dst_repeated_hour", "zone"]
    num = df[val].mul(w, axis=0)
    den = df[val].notna().mul(w, axis=0)
    g_num = num.groupby([df[k] for k in keys]).sum(min_count=1)
    g_den = den.groupby([df[k] for k in keys]).sum()
    out = (g_num / g_den.replace(0, np.nan)).astype("float32")
    extra = df.groupby(keys).agg(valid_time_utc=("valid_time_utc", "first"),
                                 issued_utc_max=("issued_utc_max", "max"))
    return extra.join(out).reset_index()


# ----------------------------------------------------------------------------- verification
def verify_semantics(model: str = "gfs", valid_start: str | None = None, n_days: int = 2,
                     lat: float = 32.9, lon: float = -97.04) -> pd.DataFrame:
    """Compare previous_day2 temperature against individual runs from the Single Runs API
    (recent weeks only). Returns, per valid hour, the lead hours of runs whose value matches and
    whether the expected run floor_cycle(t - 48h) is among them."""
    spec = MODELS[model]
    if valid_start is None:
        valid_start = (date.today() - timedelta(days=7)).isoformat()
    vs = pd.Timestamp(valid_start, tz="UTC")
    ve = vs + pd.Timedelta(days=n_days - 1)
    pr = requests.get(API_URL, params=dict(
        latitude=lat, longitude=lon, hourly="temperature_2m_previous_day2", models=spec.om_model,
        start_date=vs.date().isoformat(), end_date=ve.date().isoformat(), timezone="GMT"), timeout=60).json()
    d2 = pd.Series(pr["hourly"]["temperature_2m_previous_day2"],
                   index=pd.to_datetime(pr["hourly"]["time"], utc=True))
    runs = {}
    for r in pd.date_range(vs - pd.Timedelta(days=3), ve, freq=f"{spec.cycle_h}h"):
        resp = requests.get(SINGLE_RUN_URL, params=dict(
            latitude=lat, longitude=lon, hourly="temperature_2m", models=spec.om_model,
            run=r.strftime("%Y-%m-%dT%H:%M"), forecast_hours=120, timezone="GMT"), timeout=60)
        try:
            j = resp.json()
        except ValueError:
            continue
        if "hourly" in j:
            runs[r] = pd.Series(j["hourly"]["temperature_2m"], index=pd.to_datetime(j["hourly"]["time"], utc=True))
    rows = []
    for t, v in d2.items():
        leads = sorted((t - r) / pd.Timedelta(hours=1) for r, s in runs.items()
                       if t in s.index and s[t] is not None and v is not None and abs(s[t] - v) < 0.051)
        exp = (t - pd.Timedelta(hours=48)).floor(f"{spec.cycle_h}h")
        rows.append({"valid_utc": t, "day2": v, "matching_leads_h": leads,
                     "expected_run": exp, "expected_matches": (t - exp) / pd.Timedelta(hours=1) in leads,
                     "on_native_grid": t.hour % spec.step_h == 0})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-08-31")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--refresh", action="store_true", help="ignore cached raw JSON")
    ap.add_argument("--verify", action="store_true", help="check day2 semantics vs Single Runs API")
    ap.add_argument("--out", default=str(PARQUET_PATH), help="output parquet path")
    args = ap.parse_args(argv)
    models = args.models.split(",")
    if args.verify:
        for m in models:
            v = verify_semantics(m)
            g = v[v["on_native_grid"]]
            print(f"{m}: expected run matches {int(g['expected_matches'].sum())}/{len(g)} native-grid hours, "
                  f"{int(v['expected_matches'].sum())}/{len(v)} all hours")
        return
    df = build(date.fromisoformat(args.start), date.fromisoformat(args.end), models, args.refresh)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df):,} size={out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
