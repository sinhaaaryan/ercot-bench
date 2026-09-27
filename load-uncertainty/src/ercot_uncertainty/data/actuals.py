"""Actual hourly load by ERCOT weather zone (labels).

Reads ``load_hourly`` from the grid-analyst DuckDB (ercot_uncertainty.paths.ERCOT_DUCKDB, read-only) and writes
``data/parquet/actual_load.parquet``: operating_date, hour_ending, zone, actual_mw, plus
``dst_repeated_hour`` and ``interval_start_utc``.

DST convention (ERCOT's, unchanged from load_hourly / ERCOT native files):
- spring-forward day has 23 rows: HE 1, 2, 4, ..., 24 (HE 3 does not exist);
- fall-back day has 25 rows: HE 2 appears twice; the second (01:00-02:00 CST) has
  ``dst_repeated_hour=True``.
So (operating_date, hour_ending, zone) is NOT unique on fall-back days; join on
(operating_date, hour_ending, dst_repeated_hour, zone). The NP3-565-CD vintages
(lf_vintage.parquet) and EIA-930 (eia930_df.parquet) use the same convention and flag.

``ercot_total`` is renamed to ``system_total`` (it equals the sum of the 8 zones to <1e-5 MW).

CLI::

    uv run python -m ercot_uncertainty.data.actuals
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

from ercot_uncertainty.paths import ERCOT_DUCKDB

log = logging.getLogger("actuals")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DUCKDB_PATH = ERCOT_DUCKDB
PARQUET_PATH = PROJECT_ROOT / "data" / "parquet" / "actual_load.parquet"

ZONE_COLS = {
    "coast_mw": "coast", "east_mw": "east", "far_west_mw": "far_west", "north_mw": "north",
    "north_central_mw": "north_central", "south_mw": "south", "south_central_mw": "south_central",
    "west_mw": "west", "ercot_total_mw": "system_total",
}
COLUMNS = ["operating_date", "hour_ending", "zone", "actual_mw", "dst_repeated_hour",
           "interval_start_utc"]


def build(db_path: Path = DUCKDB_PATH, out: Path = PARQUET_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        wide = con.execute(
            "SELECT delivery_date, hour_ending, dst_repeated_hour, interval_start_utc, "
            + ", ".join(ZONE_COLS) + " FROM load_hourly ORDER BY interval_start_utc"
        ).df()
    finally:
        con.close()
    long = wide.melt(
        id_vars=["delivery_date", "hour_ending", "dst_repeated_hour", "interval_start_utc"],
        value_vars=list(ZONE_COLS), var_name="zone", value_name="actual_mw",
    )
    long["zone"] = long["zone"].map(ZONE_COLS)
    long["operating_date"] = pd.to_datetime(long["delivery_date"]).dt.date
    long["hour_ending"] = long["hour_ending"].astype("int16")
    long["dst_repeated_hour"] = long["dst_repeated_hour"].fillna(False).astype(bool)
    long["interval_start_utc"] = (
        pd.to_datetime(long["interval_start_utc"]).dt.tz_localize("UTC").astype("datetime64[ns, UTC]")
    )
    df = long[COLUMNS].sort_values(["operating_date", "zone", "interval_start_utc"])
    df = df.reset_index(drop=True)
    key = ["operating_date", "hour_ending", "dst_repeated_hour", "zone"]
    assert not df.duplicated(key).any(), "duplicate keys in load_hourly"
    log.info("rows=%d dates=%d (%s..%s) null actual_mw=%d", len(df), df.operating_date.nunique(),
             df.operating_date.min(), df.operating_date.max(), int(df.actual_mw.isna().sum()))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("wrote %s", out)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ERCOT actual load by weather zone -> parquet")
    ap.add_argument("--db", type=Path, default=DUCKDB_PATH)
    ap.add_argument("--out", type=Path, default=PARQUET_PATH)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(a.db, a.out)


if __name__ == "__main__":
    main()
