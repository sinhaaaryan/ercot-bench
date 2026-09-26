"""DuckDB database: build from parquet, and read-only connections."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import duckdb

from ercot_bench.config import Config

log = logging.getLogger("ercot_bench.db")

# table name -> (sort key columns). Column order/types come from the parquet files.
TABLES = {
    "rt_spp_15min": "settlement_point, interval_start_utc",
    "da_spp_hourly": "settlement_point, interval_start_utc",
    "load_hourly": "interval_start_utc",
    "fuel_mix_15min": "fuel, interval_start_utc",
    "as_prices_dam_hourly": "service, interval_start_utc",
    "wind_forecast_hourly": "interval_start_utc",
    "solar_forecast_hourly": "interval_start_utc",
    "load_forecast_hourly": "interval_start_utc",
}

# Explicit column order so the schema doc and the DB always agree.
COLUMN_ORDER = {
    "rt_spp_15min": ["settlement_point", "settlement_point_type", "delivery_date", "hour_ending", "interval_in_hour",
                     "dst_repeated_hour", "interval_start_local", "interval_end_local", "interval_start_utc",
                     "interval_end_utc", "price_usd_per_mwh"],
    "da_spp_hourly": ["settlement_point", "settlement_point_type", "delivery_date", "hour_ending", "dst_repeated_hour",
                      "interval_start_local", "interval_end_local", "interval_start_utc", "interval_end_utc",
                      "price_usd_per_mwh"],
    "load_hourly": ["delivery_date", "hour_ending", "dst_repeated_hour", "interval_start_local", "interval_end_local",
                    "interval_start_utc", "interval_end_utc", "coast_mw", "east_mw", "far_west_mw", "north_mw",
                    "north_central_mw", "south_mw", "south_central_mw", "west_mw", "ercot_total_mw"],
    "fuel_mix_15min": ["fuel", "delivery_date", "hour_ending", "interval_in_hour", "dst_repeated_hour",
                       "interval_start_local", "interval_end_local", "interval_start_utc", "interval_end_utc",
                       "generation_mw", "energy_mwh", "settlement_type"],
    "as_prices_dam_hourly": ["service", "delivery_date", "hour_ending", "dst_repeated_hour", "interval_start_local",
                             "interval_end_local", "interval_start_utc", "interval_end_utc", "price_usd_per_mw"],
    "wind_forecast_hourly": ["delivery_date", "hour_ending", "dst_repeated_hour", "interval_start_local",
                             "interval_end_local", "interval_start_utc", "interval_end_utc", "actual_mw",
                             "forecast_mw", "forecast_publish_time_local"],
    "solar_forecast_hourly": ["delivery_date", "hour_ending", "dst_repeated_hour", "interval_start_local",
                              "interval_end_local", "interval_start_utc", "interval_end_utc", "actual_mw",
                              "forecast_mw", "forecast_publish_time_local"],
    "load_forecast_hourly": ["delivery_date", "hour_ending", "dst_repeated_hour", "interval_start_local",
                             "interval_end_local", "interval_start_utc", "interval_end_utc", "forecast_mw",
                             "forecast_publish_time_local"],
}


def build_database(cfg: Config) -> list[str]:
    """(Re)create data/ercot.duckdb from data/parquet/<table>/*.parquet. Atomic swap."""
    tmp = cfg.db_path.with_suffix(".building.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    built = []
    for table, order in TABLES.items():
        files = sorted((cfg.parquet_dir / table).glob("*.parquet"))
        if not files:
            log.warning("no parquet for %s; table not created", table)
            continue
        cols = ", ".join(
            f"CAST({c} AS TIMESTAMP) AS {c}" if c.startswith("interval_") and c != "interval_in_hour"
            or c.endswith("_time_local") else c
            for c in COLUMN_ORDER[table])
        file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        con.execute(f"""
            CREATE TABLE {table} AS
            SELECT {cols} FROM read_parquet({file_list}, union_by_name=true)
            ORDER BY {order}
        """)
        # parquet written by pandas keeps delivery_date as DATE, bools as BOOLEAN; enforce anyway
        con.execute(f"ALTER TABLE {table} ALTER delivery_date TYPE DATE")
        built.append(table)
    con.close()
    cfg.db_path.unlink(missing_ok=True)
    tmp.rename(cfg.db_path)
    return built


_BASE: dict[str, duckdb.DuckDBPyConnection] = {}
_LOCK = threading.Lock()


def _base_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    key = str(Path(db_path).resolve())
    with _LOCK:
        con = _BASE.get(key)
        if con is None:
            # Cap DuckDB per process: its default is 80% of host RAM, and every env worker / eval process opens
            # its own connection. Override with ERCOT_DUCKDB_MEMORY_LIMIT / ERCOT_DUCKDB_THREADS.
            con = duckdb.connect(key, read_only=True, config={
                "enable_external_access": False,
                "memory_limit": os.getenv("ERCOT_DUCKDB_MEMORY_LIMIT", "1GB"),
                "threads": int(os.getenv("ERCOT_DUCKDB_THREADS", "2")),
            })
            _BASE[key] = con
        return con


def connect_readonly(db_path: Path | str) -> duckdb.DuckDBPyConnection:
    """A thread-local cursor on a shared read-only connection with external access disabled.

    duckdb connections are not thread-safe, but cursors created via .cursor() are independent
    connections to the same database and can be used concurrently from different threads.
    """
    return _base_connection(Path(db_path)).cursor()


def table_names(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1").fetchall()]
