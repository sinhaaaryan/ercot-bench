"""Family: advanced -- multi-step questions that rely on industry conventions or real optimization.

Added after Sonnet scored 100% on the first template set. These deliberately give less
hand-holding: the question names the concept (5x16 block, load-weighted price, price event,
SOC-constrained battery) and the model must know or derive the computation.
"""

from __future__ import annotations

import datetime as dt
import math

from ercot_bench.tasks.templates.base import Template, as_date, is_dst_day, month_bounds, month_name, r2

FAMILY = "advanced"
TOL2 = {"abs": 0.01, "rel": 0.0}


def nerc_holidays(year: int) -> set[dt.date]:
    """NERC off-peak holidays: New Year's, Memorial Day, Independence Day, Labor Day, Thanksgiving,
    Christmas. A holiday falling on Sunday is observed Monday; Saturday holidays are not moved."""

    def nth_weekday(month: int, weekday: int, n: int) -> dt.date:
        d = dt.date(year, month, 1)
        d += dt.timedelta(days=(weekday - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)

    def last_weekday(month: int, weekday: int) -> dt.date:
        d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
        return d - dt.timedelta(days=(d.weekday() - weekday) % 7)

    fixed = [dt.date(year, 1, 1), dt.date(year, 7, 4), dt.date(year, 12, 25)]
    fixed = [d + dt.timedelta(days=1) if d.weekday() == 6 else d for d in fixed]
    return set(fixed) | {last_weekday(5, 0), nth_weekday(9, 0, 1), nth_weekday(11, 3, 4)}


def block_of(d: dt.date, he: int) -> str:
    """ERCOT/NERC blocks: 5x16 = HE7-HE22 Mon-Fri excluding NERC holidays; 2x16 = HE7-HE22 on
    weekends and NERC holidays; 7x8 = HE1-HE6 and HE23-HE24 every day."""
    if he <= 6 or he >= 23:
        return "7x8"
    if d.weekday() >= 5 or d in nerc_holidays(d.year):
        return "2x16"
    return "5x16"


BLOCK_TEXT = {
    "5x16": "5x16 on-peak block (HE7-HE22, Monday-Friday, excluding NERC holidays)",
    "2x16": "2x16 block (HE7-HE22 on weekends and NERC holidays)",
    "7x8": "7x8 off-peak block (HE1-HE6 and HE23-HE24, every day)",
}


class BlockAvgDaMonth(Template):
    id = "block_avg_da_month"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = TOL2
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        months = ctx.months("da_spp_hourly")
        if not months:
            return None
        # bias toward months with NERC holidays or DST transitions, where the block definition bites
        tricky = [m for m in months if m[5:] in ("01", "03", "05", "07", "09", "11", "12")]
        ym = self.pick(rng, tricky if tricky and rng.random() < 0.8 else months)
        return {"sp": self.pick(rng, ctx.settlement_points), "month": ym, "block": rng.choice(list(BLOCK_TEXT))}

    def render_question(self, p):
        return (f"What was the average day-ahead settlement point price at {p['sp']} over the {BLOCK_TEXT[p['block']]} "
                f"in {month_name(p['month'])}? Weight every delivered hour in the block equally. "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        rows = ctx.con.execute("SELECT delivery_date, hour_ending, price_usd_per_mwh FROM da_spp_hourly "
                               "WHERE settlement_point=? AND delivery_date BETWEEN ? AND ?", [p["sp"], a, b]).fetchall()
        vals = [v for d, he, v in rows if block_of(d, he) == p["block"]]
        return r2(sum(vals) / len(vals)) if vals else None


def best_2h_battery(prices: list[float], eff: float) -> float:
    """1 MW / 2 MWh battery, hourly actions in {-1 charge, 0 idle, +1 discharge} MW, SOC in {0,1,2} MWh,
    starts and ends empty; discharge of 1 MWh sells eff MWh. Max revenue via DP."""
    neg = -math.inf
    best = {0: 0.0, 1: neg, 2: neg}
    for p in prices:
        nxt = {0: neg, 1: neg, 2: neg}
        for soc, v in best.items():
            if v == neg:
                continue
            nxt[soc] = max(nxt[soc], v)                                  # idle
            if soc < 2:
                nxt[soc + 1] = max(nxt[soc + 1], v - p)                  # charge 1 MWh
            if soc > 0:
                nxt[soc - 1] = max(nxt[soc - 1], v + eff * p)            # discharge 1 MWh
        best = nxt
    return best[0]


class Battery2hOptimalDa(Template):
    id = "battery_2h_optimal_da"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = {"abs": 0.05, "rel": 1e-4}
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        days = ctx.days("da_spp_hourly", exclude_dst=True)
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days)),
                "eff": rng.choice([0.85, 0.9, 1.0])}

    def render_question(self, p):
        return (f"A 1 MW / 2 MWh battery at {p['sp']} starts operating day {p['date']} empty and must end it empty. "
                f"In each hour it either charges at exactly 1 MW, discharges at exactly 1 MW, or idles, and its "
                f"stored energy must stay between 0 and 2 MWh. Each MWh discharged delivers {p['eff']:.2f} MWh to "
                f"the grid (round-trip efficiency {p['eff']:.0%}). Using hourly day-ahead prices, what is the "
                f"maximum achievable net revenue for the day? Answer in dollars, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        prices = ctx.column("SELECT price_usd_per_mwh FROM da_spp_hourly WHERE settlement_point=? AND delivery_date=? "
                            "ORDER BY interval_start_utc", [p["sp"], as_date(p["date"])])
        if len(prices) != 24:
            return None
        return r2(best_2h_battery(prices, p["eff"]))


class LoadWeightedRtMonth(Template):
    id = "load_weighted_rt_month"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = TOL2
    requires = ("rt_spp_15min", "load_hourly")

    def sample_params(self, rng, ctx):
        months = ctx.months(list(self.requires))
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.hubs), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"What was the load-weighted average real-time price at {p['sp']} in {month_name(p['month'])}, "
                f"weighting each hour's real-time price by that hour's ERCOT system load? "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        return r2(ctx.scalar("""
            WITH rt AS (SELECT date_trunc('hour', interval_start_utc) h, avg(price_usd_per_mwh) p FROM rt_spp_15min
                        WHERE settlement_point = $sp AND delivery_date BETWEEN $a AND $b GROUP BY 1)
            SELECT sum(rt.p * l.ercot_total_mw) / sum(l.ercot_total_mw)
            FROM rt JOIN load_hourly l ON l.interval_start_utc = rt.h""", {"sp": p["sp"], "a": a, "b": b}))


class PriceSpikeEvents(Template):
    id = "price_spike_events"
    family = FAMILY
    difficulty = 3
    answer_type = "integer"
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        months = ctx.months("rt_spp_15min")
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months),
                "x": rng.choice([50, 75, 100, 150, 200]), "min_len": rng.choice([1, 2, 4])}

    def render_question(self, p):
        return (f"Define a price spike at {p['sp']} as a maximal run of consecutive 15-minute real-time intervals "
                f"with price at or above ${p['x']}/MWh, lasting at least {p['min_len']} interval(s). How many price "
                f"spikes started during {month_name(p['month'])}? Answer with an integer.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        # runs are computed over the whole history so a run crossing a month boundary is one event
        rows = ctx.con.execute("""
            SELECT delivery_date, price_usd_per_mwh >= $x hi FROM rt_spp_15min
            WHERE settlement_point = $sp AND delivery_date BETWEEN $a0 AND $b1 ORDER BY interval_start_utc""",
                               {"x": p["x"], "sp": p["sp"], "a0": a - dt.timedelta(days=2),
                                "b1": b + dt.timedelta(days=2)}).fetchall()
        n, run_start, run_len = 0, None, 0
        for d, hi in rows + [(None, False)]:
            if hi:
                if run_len == 0:
                    run_start = d
                run_len += 1
            else:
                if run_len >= p["min_len"] and run_start is not None and a <= run_start <= b:
                    n += 1
                run_len = 0
        return n or None


class PeakNetLoadHourPrice(Template):
    id = "peak_netload_hour_rt"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = TOL2
    requires = ("rt_spp_15min", "load_hourly", "fuel_mix_15min")

    def sample_params(self, rng, ctx):
        months = ctx.months(list(self.requires))
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"Find the hour in {month_name(p['month'])} with the highest ERCOT net load (load minus wind minus "
                f"solar). What was the hourly real-time price at {p['sp']} in that hour? "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        rows = ctx.con.execute("""
            WITH gen AS (SELECT date_trunc('hour', interval_start_utc) h,
                                avg(generation_mw) FILTER (WHERE fuel='Wind') w, avg(generation_mw) FILTER (WHERE fuel='Solar') s
                         FROM fuel_mix_15min WHERE delivery_date BETWEEN $a AND $b GROUP BY 1),
                 nl AS (SELECT l.interval_start_utc h, l.ercot_total_mw - g.w - g.s net FROM load_hourly l
                        JOIN gen g ON g.h = l.interval_start_utc WHERE l.delivery_date BETWEEN $a AND $b)
            SELECT h, net FROM nl ORDER BY net DESC LIMIT 2""", {"a": a, "b": b}).fetchall()
        if len(rows) < 2 or abs(rows[0][1] - rows[1][1]) < 1.0:
            return None
        return r2(ctx.scalar("SELECT avg(price_usd_per_mwh) FROM rt_spp_15min WHERE settlement_point=? AND "
                             "date_trunc('hour', interval_start_utc) = ?", [p["sp"], rows[0][0]]))


class RtAboveDaShare(Template):
    id = "rt_above_da_share"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = {"abs": 0.1, "rel": 0.0}
    requires = ("rt_spp_15min", "da_spp_hourly")

    def sample_params(self, rng, ctx):
        months = ctx.months(list(self.requires))
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"In what percentage of delivery hours in {month_name(p['month'])} did the hourly real-time price at "
                f"{p['sp']} exceed the day-ahead price for the same hour? Answer as a percentage (0-100), rounded to "
                f"1 decimal.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        v = ctx.scalar("""
            WITH rt AS (SELECT date_trunc('hour', interval_start_utc) h, avg(price_usd_per_mwh) p FROM rt_spp_15min
                        WHERE settlement_point=$sp AND delivery_date BETWEEN $a AND $b GROUP BY 1)
            SELECT 100.0 * avg((rt.p > d.price_usd_per_mwh)::INT) FROM da_spp_hourly d JOIN rt ON rt.h = d.interval_start_utc
            WHERE d.settlement_point=$sp AND d.delivery_date BETWEEN $a AND $b""", {"sp": p["sp"], "a": a, "b": b})
        return None if v is None else round(v, 1)


class YoyDaChange(Template):
    id = "yoy_da_change"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = {"abs": 0.1, "rel": 0.0}
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        all_months = set(ctx.__class__(ctx.con, None).months("da_spp_hourly"))
        months = [m for m in ctx.months("da_spp_hourly") if f"{int(m[:4]) - 1}{m[4:]}" in all_months]
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months)}

    def render_question(self, p):
        prev = f"{int(p['month'][:4]) - 1}{p['month'][4:]}"
        return (f"By what percentage did the average day-ahead price at {p['sp']} change in {month_name(p['month'])} "
                f"compared with {month_name(prev)}? Use (new - old) / old x 100. Answer as a percentage, rounded to "
                f"1 decimal (negative if prices fell).")

    def reference_answer(self, p, ctx):
        prev = f"{int(p['month'][:4]) - 1}{p['month'][4:]}"
        vals = []
        for ym in (p["month"], prev):
            a, b = month_bounds(ym)
            vals.append(ctx.scalar("SELECT avg(price_usd_per_mwh) FROM da_spp_hourly WHERE settlement_point=? "
                                   "AND delivery_date BETWEEN ? AND ?", [p["sp"], a, b]))
        new, old = vals
        if new is None or not old or abs(old) < 1:
            return None
        return round((new - old) / old * 100, 1)


class FallbackDay7x8Avg(Template):
    id = "fallback_day_7x8_rt"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = TOL2
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        days = [d for d in ctx.days("rt_spp_15min") if is_dst_day(d)]
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days))}

    def render_question(self, p):
        return (f"What was the average real-time price at {p['sp']} over the off-peak hours HE1-HE6 and HE23-HE24 of "
                f"operating day {p['date']}? Include every delivered 15-minute interval in those hours. "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        return r2(ctx.scalar("SELECT avg(price_usd_per_mwh) FROM rt_spp_15min WHERE settlement_point=? AND "
                             "delivery_date=? AND (hour_ending <= 6 OR hour_ending >= 23)", [p["sp"], as_date(p["date"])]))


TEMPLATES = [BlockAvgDaMonth(), Battery2hOptimalDa(), LoadWeightedRtMonth(), PriceSpikeEvents(),
             PeakNetLoadHourPrice(), RtAboveDaShare(), YoyDaChange(), FallbackDay7x8Avg()]
