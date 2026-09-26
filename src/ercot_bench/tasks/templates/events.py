"""Family: events -- threshold counts, peak timestamps, consecutive runs, day lists."""

from __future__ import annotations

from ercot_bench.tasks.templates.base import Template, month_bounds, month_name

FAMILY = "events"
THRESHOLDS = [25, 50, 75, 100, 150, 200, 300, 500, 1000]


class CountRtAboveMonth(Template):
    id = "count_rt_above_month"
    family = FAMILY
    difficulty = 1
    answer_type = "integer"
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        months = ctx.months("rt_spp_15min")
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months), "x": rng.choice(THRESHOLDS)}

    def render_question(self, p):
        return (f"How many 15-minute real-time intervals in {month_name(p['month'])} had a settlement point price at "
                f"{p['sp']} strictly above ${p['x']}/MWh? Answer with an integer.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        n = ctx.scalar("SELECT count(*) FROM rt_spp_15min WHERE settlement_point=? AND delivery_date BETWEEN ? AND ? "
                       "AND price_usd_per_mwh > ?", [p["sp"], a, b, p["x"]])
        return n if n else None  # zero counts are too easy to guess; resample


class CountDaNegativeMonth(Template):
    id = "count_da_negative_month"
    family = FAMILY
    difficulty = 1
    answer_type = "integer"
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        months = ctx.months("da_spp_hourly")
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"How many hours in {month_name(p['month'])} had a negative day-ahead settlement point price at "
                f"{p['sp']}? Answer with an integer.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        n = ctx.scalar("SELECT count(*) FROM da_spp_hourly WHERE settlement_point=? AND delivery_date BETWEEN ? AND ? "
                       "AND price_usd_per_mwh < 0", [p["sp"], a, b])
        return n if n else None


class LongestRunAbove(Template):
    id = "longest_run_above"
    family = FAMILY
    difficulty = 3
    answer_type = "integer"
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        months = ctx.months("rt_spp_15min")
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months), "x": rng.choice(THRESHOLDS[:6])}

    def render_question(self, p):
        return (f"During {month_name(p['month'])}, what was the longest run of consecutive 15-minute real-time intervals "
                f"in which the settlement point price at {p['sp']} was at or above ${p['x']}/MWh? Answer with the number "
                f"of intervals in the longest run (an integer).")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        n = ctx.scalar("""
            WITH x AS (SELECT interval_start_utc, price_usd_per_mwh >= $x AS hi,
                              row_number() OVER (ORDER BY interval_start_utc)
                            - row_number() OVER (PARTITION BY price_usd_per_mwh >= $x ORDER BY interval_start_utc) grp
                       FROM rt_spp_15min WHERE settlement_point = $sp AND delivery_date BETWEEN $a AND $b)
            SELECT max(c) FROM (SELECT grp, count(*) c FROM x WHERE hi GROUP BY grp)""",
                       {"x": p["x"], "sp": p["sp"], "a": a, "b": b})
        return n if n else None


class PeakRtLocalMonth(Template):
    id = "peak_rt_local_month"
    family = FAMILY
    difficulty = 2
    answer_type = "timestamp"
    answer_timezone = "America/Chicago"
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        months = ctx.months("rt_spp_15min")
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"When did the highest 15-minute real-time settlement point price at {p['sp']} occur in "
                f"{month_name(p['month'])}? If tied, take the earliest. Give the interval start in Central Prevailing "
                f"Time (local wall-clock), formatted YYYY-MM-DD HH:MM.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        row = ctx.con.execute("""
            SELECT interval_start_local, delivery_date, hour_ending FROM rt_spp_15min
            WHERE settlement_point=? AND delivery_date BETWEEN ? AND ?
            ORDER BY price_usd_per_mwh DESC, interval_start_utc LIMIT 1""", [p["sp"], a, b]).fetchone()
        if row is None:
            return None
        ts, d, he = row
        if d.month == 11 and d.day <= 7 and d.weekday() == 6 and he == 2:
            return None  # local time ambiguous on the fall-back day's repeated hour
        return ts.strftime("%Y-%m-%d %H:%M")


class DaysRtExceededList(Template):
    id = "days_rt_exceeded_list"
    family = FAMILY
    difficulty = 2
    answer_type = "list"
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        months = ctx.months("rt_spp_15min")
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months), "x": rng.choice(THRESHOLDS[2:])}

    def render_question(self, p):
        return (f"List every operating day in {month_name(p['month'])} on which the real-time settlement point price at "
                f"{p['sp']} exceeded ${p['x']}/MWh in at least one 15-minute interval. Return one row per date "
                f"(YYYY-MM-DD), in any order.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        days = ctx.column("SELECT DISTINCT delivery_date FROM rt_spp_15min WHERE settlement_point=? "
                          "AND delivery_date BETWEEN ? AND ? AND price_usd_per_mwh > ? ORDER BY 1", [p["sp"], a, b, p["x"]])
        if not 1 <= len(days) <= 12:
            return None
        return [str(d) for d in days]


TEMPLATES = [CountRtAboveMonth(), CountDaNegativeMonth(), LongestRunAbove(), PeakRtLocalMonth(), DaysRtExceededList()]
