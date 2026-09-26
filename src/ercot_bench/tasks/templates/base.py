"""Template base class and the sampling context shared by all templates."""

from __future__ import annotations

import calendar
import datetime as dt
import random
from functools import cached_property
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import duckdb

from ercot_bench.config import LOCAL_TZ

TZ = ZoneInfo(LOCAL_TZ)

# per-hour interval count and entity column of each table
TABLE_GRAIN = {
    "rt_spp_15min": (4, "settlement_point"),
    "da_spp_hourly": (1, "settlement_point"),
    "load_hourly": (1, None),
    "fuel_mix_15min": (4, "fuel"),
    "as_prices_dam_hourly": (1, "service"),
    "wind_forecast_hourly": (1, None),
    "solar_forecast_hourly": (1, None),
    "load_forecast_hourly": (1, None),
}


def hours_in_day(d: dt.date) -> int:
    a = dt.datetime(d.year, d.month, d.day, tzinfo=TZ)
    nd = d + dt.timedelta(days=1)
    b = dt.datetime(nd.year, nd.month, nd.day, tzinfo=TZ)
    return round((b.timestamp() - a.timestamp()) / 3600)


def is_dst_day(d: dt.date) -> bool:
    return hours_in_day(d) != 24


def month_bounds(ym: str) -> tuple[dt.date, dt.date]:
    y, m = map(int, ym.split("-"))
    first = dt.date(y, m, 1)
    last = dt.date(y, m, calendar.monthrange(y, m)[1])
    return first, last


def month_name(ym: str) -> str:
    first, _ = month_bounds(ym)
    return first.strftime("%B %Y")


class Ctx:
    """Sampling context: DB connection + which dates/months are complete, restricted to `years`."""

    def __init__(self, con: duckdb.DuckDBPyConnection, years: set[int] | None = None):
        self.con = con
        self.years = years
        self._complete: dict[str, list[dt.date]] = {}

    @cached_property
    def tables(self) -> set[str]:
        return {r[0] for r in self.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()}

    @cached_property
    def settlement_points(self) -> list[str]:
        return [r[0] for r in self.con.execute(
            "SELECT DISTINCT settlement_point FROM da_spp_hourly ORDER BY 1").fetchall()]

    @property
    def hubs(self) -> list[str]:
        return [s for s in self.settlement_points if s.startswith("HB_")]

    @property
    def load_zones(self) -> list[str]:
        return [s for s in self.settlement_points if s.startswith("LZ_")]

    def _complete_days_all_years(self, table: str) -> list[dt.date]:
        if table not in self._complete:
            per_hour, ent = TABLE_GRAIN[table]
            n_ent = self.con.execute(f"SELECT count(DISTINCT {ent}) FROM {table}").fetchone()[0] if ent else 1
            rows = self.con.execute(f"SELECT delivery_date, count(*) FROM {table} GROUP BY 1 ORDER BY 1").fetchall()
            self._complete[table] = [d for d, c in rows if c == per_hour * hours_in_day(d) * n_ent]
        return self._complete[table]

    def days(self, tables: str | list[str], exclude_dst: bool = False) -> list[dt.date]:
        tables = [tables] if isinstance(tables, str) else tables
        if not all(t in self.tables for t in tables):
            return []
        sets = [set(self._complete_days_all_years(t)) for t in tables]
        days = sorted(set.intersection(*sets))
        if self.years is not None:
            days = [d for d in days if d.year in self.years]
        if exclude_dst:
            days = [d for d in days if not is_dst_day(d)]
        return days

    def dst_days(self, tables: str | list[str], kind: str | None = None) -> list[dt.date]:
        """kind: 'spring' | 'fall' | None (both)."""
        out = [d for d in self.days(tables) if is_dst_day(d)]
        if kind == "spring":
            out = [d for d in out if hours_in_day(d) == 23]
        elif kind == "fall":
            out = [d for d in out if hours_in_day(d) == 25]
        return out

    def months(self, tables: str | list[str]) -> list[str]:
        days = set(self.days(tables))
        months = sorted({d.strftime("%Y-%m") for d in days})
        out = []
        for ym in months:
            first, last = month_bounds(ym)
            n = (last - first).days + 1
            if all(first + dt.timedelta(days=i) in days for i in range(n)):
                out.append(ym)
        return out

    # query helpers
    def scalar(self, sql: str, params: list | None = None) -> Any:
        row = self.con.execute(sql, params or []).fetchone()
        return None if row is None else row[0]

    def column(self, sql: str, params: list | None = None) -> list:
        return [r[0] for r in self.con.execute(sql, params or []).fetchall()]


class Template:
    """One question template. Subclasses set the class attributes and implement 3 methods."""

    id: ClassVar[str]
    family: ClassVar[str]
    difficulty: ClassVar[int]
    answer_type: ClassVar[str]
    tolerance: ClassVar[dict] = {"abs": 0.0, "rel": 0.0}
    answer_timezone: ClassVar[str | None] = None
    ordered: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ()

    def available(self, ctx: Ctx) -> bool:
        return all(t in ctx.tables for t in self.requires)

    def sample_params(self, rng: random.Random, ctx: Ctx) -> dict | None:
        """Return params, or None if nothing can be sampled in this context (e.g. no dates)."""
        raise NotImplementedError

    def render_question(self, params: dict) -> str:
        raise NotImplementedError

    def reference_answer(self, params: dict, ctx: Ctx) -> Any:
        """Ground truth from our own SQL/Python. Return None if the sample is degenerate."""
        raise NotImplementedError

    # ---- helpers
    @staticmethod
    def pick(rng: random.Random, xs: list):
        return rng.choice(xs) if xs else None


def as_date(s: str | dt.date) -> dt.date:
    return s if isinstance(s, dt.date) else dt.date.fromisoformat(s)


def r2(x) -> float | None:
    return None if x is None else round(float(x), 2)
