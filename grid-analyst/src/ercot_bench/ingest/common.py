"""Shared time normalization for every ERCOT table.

Conventions (documented in schema_doc.py as well):
- interval_start_utc / interval_end_utc: naive TIMESTAMP in UTC. Unambiguous; use for joins.
- interval_start_local / interval_end_local: naive wall-clock TIMESTAMP in Central Prevailing Time
  (America/Chicago). Ambiguous during the fall-back repeated hour -> see dst_repeated_hour.
- delivery_date: ERCOT operating day (local date of interval start).
- hour_ending: ERCOT hour-ending label 1..24 in local time. HE1 = 00:00-01:00 local.
  On the spring-forward day HE3 does not exist (23 hours). On the fall-back day HE2 occurs twice;
  the second occurrence has dst_repeated_hour = TRUE (ERCOT's DSTFlag = 'Y'), giving 25 hours.
- interval_in_hour (15-minute tables only): 1..4 within the hour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ercot_bench.config import LOCAL_TZ

log = logging.getLogger("ercot_bench.ingest")


def add_time_columns(df: pd.DataFrame, start_col: str, minutes: int, fifteen_min: bool) -> pd.DataFrame:
    """Given a tz-aware interval-start column, add all normalized time columns."""
    start = pd.to_datetime(df[start_col])
    if start.dt.tz is None:
        raise ValueError(f"{start_col} must be tz-aware")
    start_local = start.dt.tz_convert(LOCAL_TZ)
    start_utc = start.dt.tz_convert("UTC")
    end_utc = start_utc + pd.Timedelta(minutes=minutes)
    end_local = end_utc.dt.tz_convert(LOCAL_TZ)

    out = pd.DataFrame(index=df.index)
    out["interval_start_utc"] = start_utc.dt.tz_localize(None)
    out["interval_end_utc"] = end_utc.dt.tz_localize(None)
    out["interval_start_local"] = start_local.dt.tz_localize(None)
    out["interval_end_local"] = end_local.dt.tz_localize(None)
    out["delivery_date"] = start_local.dt.date
    out["hour_ending"] = (start_local.dt.hour + 1).astype("int16")
    offset_h = start_local.map(lambda t: t.utcoffset().total_seconds() / 3600)
    # Repeated hour: 01:00-02:00 local in standard time (UTC-6) on a day that also has it in DST.
    is_fallback_day = start_local.dt.month.eq(11) & start_local.dt.day.le(7) & start_local.dt.dayofweek.eq(6)
    out["dst_repeated_hour"] = is_fallback_day & start_local.dt.hour.eq(1) & offset_h.eq(-6.0)
    if fifteen_min:
        out["interval_in_hour"] = (start_local.dt.minute // 15 + 1).astype("int16")
    return out


def write_monthly_parquet(df: pd.DataFrame, table_dir: Path, overwrite_months: set[str] | None = None) -> list[str]:
    """Partition by local month (YYYY-MM from interval_start_local) and write one parquet per month.

    Existing month files are kept unless listed in overwrite_months (used for the current,
    still-growing month). Returns the months written.
    """
    table_dir.mkdir(parents=True, exist_ok=True)
    months = df["interval_start_local"].dt.strftime("%Y-%m")
    written = []
    for m, part in df.groupby(months):
        path = table_dir / f"{m}.parquet"
        if path.exists() and not (overwrite_months and m in overwrite_months):
            continue
        part.reset_index(drop=True).to_parquet(path, index=False)
        written.append(m)
    return written


def current_and_recent_months(n: int = 2) -> set[str]:
    now = pd.Timestamp.now(tz=LOCAL_TZ)
    return {(now - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(n)}
