"""Shared ERCOT time conventions (operating day, hour ending, cutoff, zone keys).

Hour-ending convention (matches ERCOT native files and /hackathon/ercot-bench load_hourly):
take the interval START in America/Chicago; ``operating_date`` is its local date and
``hour_ending = start_local.hour + 1``.

- Spring-forward day: HEs are 1, 2, 4, ..., 24 (23 rows). The hour 07:00-08:00 UTC
  (01:00 CST - 03:00 CDT) is HE 2; HE 3 does not exist.
- Fall-back day: HEs are 1, 2, 2*, 3, ..., 24 (25 rows). The second HE 2 (01:00-02:00 CST) has
  ``dst_repeated_hour=True`` (ERCOT's ``DSTFlag=Y``).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

TZ = "America/Chicago"

ZONES = [
    "coast", "east", "far_west", "north", "north_central", "south", "south_central", "west",
    "system_total",
]

# ERCOT file column -> contract zone key
ERCOT_ZONE_COLUMNS = {
    "Coast": "coast",
    "East": "east",
    "FarWest": "far_west",
    "North": "north",
    "NorthCentral": "north_central",
    "SouthCentral": "south_central",
    "Southern": "south",
    "West": "west",
    "SystemTotal": "system_total",
}


def cutoff_utc(operating_date: date | str | pd.Timestamp) -> pd.Timestamp:
    """CUTOFF(D) = 10:00 America/Chicago on D-1, returned as a tz-aware UTC Timestamp."""
    d = pd.Timestamp(operating_date).normalize().tz_localize(None)
    local = (d - pd.Timedelta(days=1)) + pd.Timedelta(hours=10)
    return local.tz_localize(TZ).tz_convert("UTC")


def he_from_interval_start_utc(start_utc: pd.Series) -> pd.DataFrame:
    """Map hourly interval starts (tz-aware UTC) to ERCOT operating_date / hour_ending /
    dst_repeated_hour. See module docstring for the convention."""
    s = pd.to_datetime(start_utc, utc=True)
    local = s.dt.tz_convert(TZ)
    # A repeated local hour is the one in standard time whose wall-clock hour also occurred in
    # daylight time the same night: UTC offset -6h while one hour earlier (UTC) was -5h and the
    # local hour is the same.
    offset = local.map(lambda t: t.utcoffset())
    prev_local = (s - pd.Timedelta(hours=1)).dt.tz_convert(TZ)
    repeated = (offset == pd.Timedelta(hours=-6)) & (
        prev_local.map(lambda t: t.utcoffset()) == pd.Timedelta(hours=-5)
    ) & (prev_local.dt.hour == local.dt.hour)
    return pd.DataFrame(
        {
            "operating_date": local.dt.date,
            "hour_ending": (local.dt.hour + 1).astype("int16"),
            "dst_repeated_hour": repeated.astype(bool),
        },
        index=start_utc.index,
    )
