"""EIA-930 day-ahead demand forecast for ERCOT (no-key fallback target).

Downloads EIA-930 six-month BALANCE files (no API key)::

    https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_<YYYY>_<Jan_Jun|Jul_Dec>.csv

caches them under ``data/raw/eia930/``, keeps the ``ERCO`` rows and writes
``data/parquet/eia930_df.parquet`` per docs/DATA_CONTRACT.md:

operating_date, hour_ending, zone="system_total", model="EIA930_DF", in_use=True, forecast_mw,
publish_time_utc=NaT, plus ``dst_repeated_hour`` (ERCOT convention), ``demand_mw`` (EIA's
reported demand, for sanity checks), ``interval_end_utc`` and ``inferred_publish_time_utc``.

Time handling: only the "UTC Time at End of Hour" column is trusted. interval start = end - 1h,
mapped to ERCOT operating_date / hour_ending with ``ercot_time.he_from_interval_start_utc``
(spring-forward day HE 1,2,4..24; fall-back day repeats HE 2 with dst_repeated_hour=True).

Vintage caveat (checked 2026-09-26 against every NP3-565-CD post on the public MIS, 8 days):
EIA's ERCO "Demand Forecast (MW)" equals ERCOT's in-use SystemTotal forecast from the post at
**14:30 CPT on D-1** (MAE 0.3 MW = rounding; the 09:30 pre-cutoff post differs by ~80 MW MAE,
other hours by 20-350 MW). So this forecast is issued ~4.5 h AFTER the 10:00 cutoff. As a
*target* (actual - forecast) that is a mild mismatch with the "dam" vintage; it must never be
used as an input feature for D. ``publish_time_utc`` stays NaT per the contract;
``inferred_publish_time_utc`` holds 14:30 America/Chicago on D-1 (verified only for recent days;
assumed for history).

CLI::

    uv run python -m ercot_uncertainty.data.eia930 --start-year 2023
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from ercot_uncertainty.data.ercot_time import he_from_interval_start_utc

log = logging.getLogger("eia930")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "eia930"
PARQUET_PATH = PROJECT_ROOT / "data" / "parquet" / "eia930_df.parquet"
URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{tag}.csv"

BA = "ERCO"
COLUMNS = [
    "operating_date", "hour_ending", "zone", "model", "in_use", "forecast_mw", "publish_time_utc",
    "dst_repeated_hour", "demand_mw", "interval_end_utc", "inferred_publish_time_utc",
]


def half_year_tags(start_year: int, today: date | None = None) -> list[str]:
    today = today or date.today()
    tags = []
    for y in range(start_year, today.year + 1):
        tags.append(f"{y}_Jan_Jun")
        if y < today.year or today.month >= 7:
            tags.append(f"{y}_Jul_Dec")
    return tags


def fetch(tag: str, refresh: bool = False, session: requests.Session | None = None) -> Path | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"EIA930_BALANCE_{tag}.csv"
    if path.exists() and not refresh:
        # Re-download files that may still be growing (modified < 45 days after period end).
        y, m1, _ = tag.split("_")
        period_end = date(int(y), 6, 30) if m1 == "Jan" else date(int(y), 12, 31)
        mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
        if mtime > period_end + timedelta(days=45):
            return path
    s = session or requests.Session()
    url = URL.format(tag=tag)
    for attempt in range(4):
        try:
            r = s.get(url, timeout=(10, 600))
            if r.status_code == 404:
                log.info("%s not published", tag)
                return None
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(path)
            log.info("downloaded %s (%.1f MB)", tag, len(r.content) / 1e6)
            return path
        except requests.RequestException as e:
            log.warning("%s: %r (attempt %d)", tag, e, attempt + 1)
            time.sleep(5 * 2**attempt)
    raise RuntimeError(f"failed to download {url}")


def read_erco(path: Path) -> pd.DataFrame:
    cols = ["Balancing Authority", "UTC Time at End of Hour", "Demand Forecast (MW)",
            "Demand (MW)"]
    df = pd.read_csv(path, usecols=cols, dtype=str)
    df = df[df["Balancing Authority"].str.strip() == BA]
    num = lambda s: pd.to_numeric(s.str.replace(",", ""), errors="coerce")  # noqa: E731
    return pd.DataFrame({
        "interval_end_utc": pd.to_datetime(df["UTC Time at End of Hour"],
                                           format="%m/%d/%Y %I:%M:%S %p", utc=True),
        "forecast_mw": num(df["Demand Forecast (MW)"]).astype(float),
        "demand_mw": num(df["Demand (MW)"]).astype(float),
    })


def build(start_year: int = 2023, refresh: bool = False, out: Path = PARQUET_PATH) -> pd.DataFrame:
    s = requests.Session()
    parts = []
    for tag in half_year_tags(start_year):
        p = fetch(tag, refresh=refresh, session=s)
        if p is not None:
            parts.append(read_erco(p))
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.drop_duplicates("interval_end_utc", keep="last").sort_values("interval_end_utc")
    he = he_from_interval_start_utc(raw["interval_end_utc"] - pd.Timedelta(hours=1))
    df = pd.concat([raw, he], axis=1)
    df = df[df["operating_date"] >= date(start_year, 1, 1)]
    df["zone"] = "system_total"
    df["model"] = "EIA930_DF"
    df["in_use"] = True
    df["publish_time_utc"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    d1 = pd.to_datetime(df["operating_date"]) - pd.Timedelta(days=1) + pd.Timedelta(hours=14.5)
    df["inferred_publish_time_utc"] = (
        d1.dt.tz_localize("America/Chicago").dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    )
    df = df[COLUMNS].reset_index(drop=True)
    n_fc = df["forecast_mw"].notna().sum()
    log.info("rows=%d (%s..%s), forecast non-null=%d, demand non-null=%d", len(df),
             df["operating_date"].min(), df["operating_date"].max(), n_fc,
             df["demand_mw"].notna().sum())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %s", out)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="EIA-930 ERCO day-ahead demand forecast")
    ap.add_argument("--start-year", type=int, default=2023)
    ap.add_argument("--refresh", action="store_true", help="re-download all files")
    ap.add_argument("--out", type=Path, default=PARQUET_PATH)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(a.start_year, a.refresh, a.out)


if __name__ == "__main__":
    main()
