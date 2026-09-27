"""Concise schema description shown to models. Built from the live DB (tables/types/date
coverage) plus curated column meanings, so it never documents a column that doesn't exist."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ercot_bench.db import connect_readonly

TIME_COLS = {
    "delivery_date": "ERCOT operating day (local date of interval start)",
    "hour_ending": "ERCOT hour-ending 1..24, local time (HE1 = 00:00-01:00)",
    "interval_in_hour": "15-min interval within the hour, 1..4",
    "dst_repeated_hour": "TRUE only for the 2nd occurrence of HE2 on the fall-back day (ERCOT DSTFlag='Y')",
    "interval_start_local": "interval start, naive wall-clock Central Prevailing Time",
    "interval_end_local": "interval end, naive wall-clock Central Prevailing Time",
    "interval_start_utc": "interval start, naive UTC (unambiguous; use for joins)",
    "interval_end_utc": "interval end, naive UTC",
}

TABLE_DOCS = {
    "rt_spp_15min": (
        "Real-time settlement point prices, one row per settlement point per 15-minute interval.",
        {
            "settlement_point": "HB_* trading hubs and LZ_* load zones (ERCOT names)",
            "settlement_point_type": "'hub' or 'load_zone'",
            "price_usd_per_mwh": "RT settlement point price, $/MWh (can be negative)",
        },
    ),
    "da_spp_hourly": (
        "Day-ahead market settlement point prices, one row per settlement point per hour.",
        {
            "settlement_point": "HB_* trading hubs and LZ_* load zones (ERCOT names)",
            "settlement_point_type": "'hub' or 'load_zone'",
            "price_usd_per_mwh": "DA settlement point price, $/MWh",
        },
    ),
    "load_hourly": (
        "Actual hourly load (ERCOT native load) by weather zone, average MW over the hour.",
        {
            "coast_mw": "Coast weather zone load, MW", "east_mw": "East, MW", "far_west_mw": "Far West, MW",
            "north_mw": "North, MW", "north_central_mw": "North Central, MW", "south_mw": "South, MW",
            "south_central_mw": "South Central, MW", "west_mw": "West, MW",
            "ercot_total_mw": "ERCOT system total load, MW (sum of zones)",
        },
    ),
    "fuel_mix_15min": (
        "Actual generation by fuel type per 15-minute interval (ERCOT IntGenbyFuel report).",
        {
            "fuel": "Biomass, Coal, Gas, Gas-CC, Hydro, Nuclear, Other, Solar, Wind, WSL "
                    "(WSL = wholesale storage load: storage charging, reported as negative)",
            "generation_mw": "average MW over the 15-min interval (= energy_mwh * 4)",
            "energy_mwh": "energy in the 15-min interval, MWh",
            "settlement_type": "ERCOT settlement stage of the data (e.g. INITIAL, FINAL)",
        },
    ),
    "as_prices_dam_hourly": (
        "Day-ahead ancillary service market clearing prices for capacity (MCPC), per service per hour.",
        {
            "service": "REGUP, REGDN, RRS, NSPIN, ECRS (ECRS price is NULL before ECRS launched in June 2023)",
            "price_usd_per_mw": "clearing price, $/MW per hour of capacity",
        },
    ),
    "wind_forecast_hourly": (
        "System-wide wind: hourly actual generation and the day-ahead forecast (STWPF).",
        {
            "actual_mw": "actual wind generation, average MW over the hour",
            "forecast_mw": "STWPF forecast from the latest report published by 10:00 local the day before",
            "forecast_publish_time_local": "publish time of that forecast, local",
        },
    ),
    "solar_forecast_hourly": (
        "System-wide solar: hourly actual generation and the day-ahead forecast (STPPF).",
        {
            "actual_mw": "actual solar generation, average MW over the hour",
            "forecast_mw": "STPPF forecast from the latest report published by 10:00 local the day before",
            "forecast_publish_time_local": "publish time of that forecast, local",
        },
    ),
    "load_forecast_hourly": (
        "ERCOT system load forecast (seven-day forecast, system total), hourly.",
        {
            "forecast_mw": "forecast system load, MW, from the latest forecast published by 10:00 local the day before",
            "forecast_publish_time_local": "publish time of that forecast, local",
        },
    ),
}

CONVENTIONS = """\
Conventions (all tables):
- Times: *_utc columns are naive UTC; *_local columns are naive Central Prevailing Time (America/Chicago).
- ERCOT labels hours by hour ending (HE1..HE24). hour_ending = local hour of interval start + 1.
- DST: the spring-forward day has 23 hours (no HE3); the fall-back day has 25 hours (HE2 appears twice;
  the repeated one has dst_repeated_hour = TRUE). Local timestamps repeat on that day; UTC never does.
- Real-time prices are 15-minute; day-ahead prices, load, AS prices and forecasts are hourly.
  An hourly RT price is the simple average of that hour's four 15-minute prices.
- Units: prices $/MWh (AS: $/MW), power MW, energy MWh. 1 $/MWh = 0.1 cents/kWh.
- A "day" or "date" in a question means the ERCOT operating day (delivery_date) unless it says UTC.
- Net load = load - wind - solar (all MW)."""


STANDARD_TIME_TEXT = """\
Every table has these standard time columns (not repeated below):
  delivery_date DATE -- ERCOT operating day (local date of interval start)
  hour_ending SMALLINT -- ERCOT hour ending 1..24 in local time (HE1 = 00:00-01:00)
  interval_in_hour SMALLINT -- 15-minute tables only: interval 1..4 within the hour
  dst_repeated_hour BOOLEAN -- TRUE only for the 2nd occurrence of HE2 on the fall-back day (ERCOT DSTFlag='Y')
  interval_start_local, interval_end_local TIMESTAMP -- naive wall-clock Central Prevailing Time
  interval_start_utc, interval_end_utc TIMESTAMP -- naive UTC (unambiguous; use for joins across tables)"""


@lru_cache(maxsize=4)
def schema_doc(db_path: Path | str) -> str:
    con = connect_readonly(db_path)
    try:
        parts = ["Database: DuckDB.", STANDARD_TIME_TEXT, ""]
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1").fetchall()]
        for t in [t for t in TABLE_DOCS if t in tables]:
            desc, col_docs = TABLE_DOCS[t]
            lo, hi = con.execute(f"SELECT min(delivery_date), max(delivery_date) FROM {t}").fetchone()
            parts.append(f"TABLE {t} -- {desc} Coverage: {lo} to {hi}.")
            for name, typ in con.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    f"WHERE table_name='{t}' ORDER BY ordinal_position").fetchall():
                if name in TIME_COLS:
                    continue
                meaning = col_docs.get(name, "")
                parts.append(f"  {name} {typ}" + (f" -- {meaning}" if meaning else ""))
            if t in ("rt_spp_15min", "da_spp_hourly"):
                sps = [r[0] for r in con.execute(f"SELECT DISTINCT settlement_point FROM {t} ORDER BY 1").fetchall()]
                parts.append(f"  settlement points: {', '.join(sps)}")
            parts.append("")
        parts.append(CONVENTIONS)
        return "\n".join(parts)
    finally:
        con.close()
