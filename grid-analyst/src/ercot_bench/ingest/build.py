"""Ingest orchestration: raw pulls -> normalized monthly parquet -> (db.py) DuckDB tables."""

from __future__ import annotations

import logging

import pandas as pd

from ercot_bench.config import LOCAL_TZ, Config
from ercot_bench.ingest import sources
from ercot_bench.ingest.common import add_time_columns, current_and_recent_months, write_monthly_parquet

log = logging.getLogger("ercot_bench.ingest")

LOAD_ZONE_COLUMNS = {
    "Coast": "coast_mw", "East": "east_mw", "Far West": "far_west_mw", "North": "north_mw",
    "North Central": "north_central_mw", "South": "south_mw", "South Central": "south_central_mw",
    "West": "west_mw", "ERCOT": "ercot_total_mw",
}
AS_SERVICES = {"REGUP": "REGUP", "REGDN": "REGDN", "RRS": "RRS", "NSPIN": "NSPIN", "ECRS": "ECRS"}


def _clip(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    start = pd.Timestamp(cfg.start_date)
    df = df[df["interval_start_local"] >= start]
    if cfg.end_date:
        df = df[df["interval_start_local"] < pd.Timestamp(cfg.end_date) + pd.Timedelta(days=1)]
    return df


def normalize_spp(raw: pd.DataFrame, minutes: int) -> pd.DataFrame:
    t = add_time_columns(raw, "Interval Start", minutes, fifteen_min=minutes == 15)
    t["settlement_point"] = raw["Location"].astype(str).values
    t["settlement_point_type"] = raw["Location Type"].map({"Trading Hub": "hub", "Load Zone": "load_zone"}).values
    t["price_usd_per_mwh"] = raw["SPP"].astype(float).values
    return t


def normalize_load(raw: pd.DataFrame) -> pd.DataFrame:
    t = add_time_columns(raw, "Interval Start", 60, fifteen_min=False)
    for src, dst in LOAD_ZONE_COLUMNS.items():
        t[dst] = pd.to_numeric(raw[src], errors="coerce").values
    return t


def normalize_fuel_mix(raw: pd.DataFrame) -> pd.DataFrame:
    t = add_time_columns(raw, "interval_start", 15, fifteen_min=True)
    t["fuel"] = raw["fuel"].values
    t["generation_mw"] = (raw["energy_mwh"] * 4.0).values  # MWh per 15 min -> average MW
    t["energy_mwh"] = raw["energy_mwh"].values
    t["settlement_type"] = raw["settlement_type"].values
    return t


def normalize_as_prices(csv_paths) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    date = pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y")
    he = df["Hour Ending"].str.slice(0, 2).astype(int)
    naive_start = date + pd.to_timedelta(he - 1, unit="h")
    repeated = df["Repeated Hour Flag"].astype(str).str.strip().eq("Y")
    # ambiguous=True means "this wall-clock time is in DST"; the repeated (flag Y) hour is standard time.
    start = naive_start.dt.tz_localize(LOCAL_TZ, ambiguous=(~repeated).values, nonexistent="shift_forward")
    base = pd.DataFrame({"interval_start": start})
    t = add_time_columns(base, "interval_start", 60, fifteen_min=False)
    t["hour_ending"] = he.astype("int16").values  # keep ERCOT's native label verbatim
    long = []
    for col, svc in AS_SERVICES.items():
        if col not in df:
            continue
        x = t.copy()
        x["service"] = svc
        x["price_usd_per_mw"] = pd.to_numeric(df[col], errors="coerce").values
        long.append(x)
    return pd.concat(long, ignore_index=True)


def _latest_forecast_before(pubs: pd.DataFrame, value_col: str, cutoff_by_day: bool = True) -> pd.DataFrame:
    """For each target hour, pick the value from the latest publication issued at or before
    10:00 local on the day before the operating day (the DAM-relevant forecast)."""
    df = pubs.dropna(subset=[value_col]).copy()
    start_local = df["Interval Start"].dt.tz_convert(LOCAL_TZ)
    cutoff = start_local.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
    # only publications from the day before (00:00-10:00 local), never multi-day-ahead ones
    df = df[(df["Publish Time"] <= cutoff) & (df["Publish Time"] > cutoff - pd.Timedelta(hours=24))]
    df = df.sort_values("Publish Time").groupby("Interval Start", as_index=False).last()
    return df[["Interval Start", "Publish Time", value_col]]


def normalize_wind_solar(pubs: pd.DataFrame, kind: str) -> pd.DataFrame:
    gen_col = "GEN SYSTEM WIDE"
    fc_col = "STWPF SYSTEM WIDE" if kind == "wind" else "STPPF SYSTEM WIDE"
    fc = _latest_forecast_before(pubs, fc_col)
    act = pubs.dropna(subset=[gen_col]).sort_values("Publish Time").groupby("Interval Start", as_index=False).last()
    m = fc.merge(act[["Interval Start", gen_col]], on="Interval Start", how="inner")
    t = add_time_columns(m, "Interval Start", 60, fifteen_min=False)
    t["actual_mw"] = m[gen_col].values
    t["forecast_mw"] = m[fc_col].values
    t["forecast_publish_time_local"] = m["Publish Time"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).values
    return t


def normalize_load_forecast(pubs: pd.DataFrame) -> pd.DataFrame:
    col = "System Total"
    fc = _latest_forecast_before(pubs, col)
    t = add_time_columns(fc, "Interval Start", 60, fifteen_min=False)
    t["forecast_mw"] = fc[col].values
    t["forecast_publish_time_local"] = fc["Publish Time"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).values
    return t


def run_ingest(cfg: Config, only: list[str] | None = None, max_age_hours: float = 20) -> dict:
    years = sources.years_in_range(cfg)
    recent = current_and_recent_months(2)
    summary: dict[str, list[str]] = {}

    def want(name: str) -> bool:
        return not only or name in only

    for market, table, minutes in (("rt", "rt_spp_15min", 15), ("da", "da_spp_hourly", 60)):
        if not want(table):
            continue
        written = []
        for y in years:
            p = sources.pull_spp(cfg, market, y, max_age_hours)
            df = _clip(normalize_spp(pd.read_parquet(p), minutes), cfg)
            written += write_monthly_parquet(df, cfg.parquet_dir / table, overwrite_months=recent)
        summary[table] = written

    if want("load_hourly"):
        written = []
        for y in years:
            try:
                p = sources.pull_load(cfg, y, max_age_hours)
            except Exception as e:  # noqa: BLE001
                log.warning("native load %d unavailable: %s", y, str(e)[:200])
                continue
            df = _clip(normalize_load(pd.read_parquet(p)), cfg)
            written += write_monthly_parquet(df, cfg.parquet_dir / "load_hourly", overwrite_months=recent)
        summary["load_hourly"] = written

    if want("fuel_mix_15min"):
        written = []
        for y in years:
            p = sources.pull_fuel_mix(cfg, y, max_age_hours)
            if p is None:
                continue
            df = _clip(normalize_fuel_mix(sources.parse_fuel_mix_workbook(p)), cfg)
            written += write_monthly_parquet(df, cfg.parquet_dir / "fuel_mix_15min", overwrite_months=recent)
        summary["fuel_mix_15min"] = written

    if want("as_prices_dam_hourly"):
        paths = sources.pull_as_prices(cfg, years, max_age_hours)
        df = _clip(normalize_as_prices(paths), cfg)
        summary["as_prices_dam_hourly"] = write_monthly_parquet(
            df, cfg.parquet_dir / "as_prices_dam_hourly", overwrite_months=recent)

    if want("forecasts"):
        summary.update(ingest_forecasts(cfg))
    return summary


def ingest_forecasts(cfg: Config, days_back: int = 9) -> dict:
    """Wind/solar/load forecasts. Without ERCOT API keys only the public ~8-day MIS window exists."""
    use_api = sources.ercot_api_available()
    today = pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).normalize()
    if use_api:
        first = pd.Timestamp(cfg.start_date) - pd.Timedelta(days=1)
    else:
        first = today - pd.Timedelta(days=days_back)
        log.warning("ERCOT API keys not set: forecast tables limited to the public MIS window (%s..)", first.date())
    days = pd.date_range(first, today, freq="D")
    out = {}
    for kind in ("wind", "solar", "load"):
        paths = [p for d in days if (p := sources.pull_forecast_day(cfg, kind, d, use_api))]
        if not paths:
            continue
        pubs = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        df = normalize_load_forecast(pubs) if kind == "load" else normalize_wind_solar(pubs, kind)
        df = _clip(df, cfg)
        table = f"{kind}_forecast_hourly"
        tdir = cfg.parquet_dir / table
        # forecast tables are small; always rewrite fully
        if tdir.exists():
            for f in tdir.glob("*.parquet"):
                f.unlink()
        out[table] = write_monthly_parquet(df, tdir)
    return out
