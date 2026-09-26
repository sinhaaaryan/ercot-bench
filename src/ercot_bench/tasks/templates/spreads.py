"""Family: spreads -- DA vs RT, hub vs load zone basis, zone comparisons."""

from __future__ import annotations

from ercot_bench.tasks.templates.base import Template, as_date, month_bounds, month_name, r2

FAMILY = "spreads"
TOL2 = {"abs": 0.01, "rel": 0.0}
HUB_LZ_PAIRS = [("LZ_HOUSTON", "HB_HOUSTON"), ("LZ_NORTH", "HB_NORTH"), ("LZ_SOUTH", "HB_SOUTH"), ("LZ_WEST", "HB_WEST")]


class DaRtSpreadDay(Template):
    id = "da_rt_spread_day"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = TOL2
    requires = ("da_spp_hourly", "rt_spp_15min")

    def sample_params(self, rng, ctx):
        days = ctx.days(["da_spp_hourly", "rt_spp_15min"])
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days))}

    def render_question(self, p):
        return (f"On operating day {p['date']}, what was the DA-RT spread at {p['sp']}, defined as the average of the "
                f"hourly day-ahead prices minus the average of the 15-minute real-time prices for that day? "
                f"Answer in $/MWh, rounded to 2 decimals (negative if RT averaged higher).")

    def reference_answer(self, p, ctx):
        d = as_date(p["date"])
        return r2(ctx.scalar("""
            SELECT (SELECT avg(price_usd_per_mwh) FROM da_spp_hourly WHERE settlement_point=$1 AND delivery_date=$2)
                 - (SELECT avg(price_usd_per_mwh) FROM rt_spp_15min WHERE settlement_point=$1 AND delivery_date=$2)""",
                             [p["sp"], d]))


class HubLzBasisMonth(Template):
    id = "hub_lz_basis_month"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = TOL2
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        months = ctx.months("rt_spp_15min")
        if not months:
            return None
        lz, hub = self.pick(rng, HUB_LZ_PAIRS)
        return {"lz": lz, "hub": hub, "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"What was the average real-time basis between {p['lz']} and {p['hub']} during {month_name(p['month'])}, "
                f"defined as the average 15-minute real-time price at {p['lz']} minus the average at {p['hub']} over "
                f"the month? Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        return r2(ctx.scalar("""
            SELECT avg(price_usd_per_mwh) FILTER (WHERE settlement_point=$1)
                 - avg(price_usd_per_mwh) FILTER (WHERE settlement_point=$2)
            FROM rt_spp_15min WHERE delivery_date BETWEEN $3 AND $4""", [p["lz"], p["hub"], a, b]))


class HighestAvgLoadZoneMonth(Template):
    id = "highest_avg_lz_month"
    family = FAMILY
    difficulty = 1
    answer_type = "category"
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        months = ctx.months("da_spp_hourly")
        if not months:
            return None
        return {"month": self.pick(rng, months), "market": rng.choice(["day-ahead", "real-time"])}

    def render_question(self, p):
        return (f"Among ERCOT load zones (settlement points starting with LZ_), which had the highest average "
                f"{p['market']} settlement point price during {month_name(p['month'])}? "
                f"Answer with the settlement point name (e.g. LZ_NORTH).")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        table = "da_spp_hourly" if p["market"] == "day-ahead" else "rt_spp_15min"
        rows = ctx.con.execute(f"""
            SELECT settlement_point, avg(price_usd_per_mwh) v FROM {table}
            WHERE settlement_point LIKE 'LZ\\_%' ESCAPE '\\' AND delivery_date BETWEEN ? AND ?
            GROUP BY 1 ORDER BY v DESC LIMIT 2""", [a, b]).fetchall()
        if len(rows) < 2 or abs(rows[0][1] - rows[1][1]) < 0.005:
            return None  # ambiguous tie
        return rows[0][0]


class MaxHourlyDaRtGapHe(Template):
    id = "max_hourly_da_rt_gap_he"
    family = FAMILY
    difficulty = 3
    answer_type = "integer"
    requires = ("da_spp_hourly", "rt_spp_15min")

    def sample_params(self, rng, ctx):
        days = ctx.days(["da_spp_hourly", "rt_spp_15min"], exclude_dst=True)
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days))}

    def render_question(self, p):
        return (f"On operating day {p['date']} at {p['sp']}, for which hour ending was the absolute difference between the "
                f"day-ahead price and the hourly real-time price largest? The hourly real-time price is the simple "
                f"average of the four 15-minute prices in the hour. If tied, give the earliest hour. "
                f"Answer with the hour-ending number (1-24).")

    def reference_answer(self, p, ctx):
        rows = ctx.con.execute("""
            WITH rt AS (SELECT hour_ending, avg(price_usd_per_mwh) p FROM rt_spp_15min
                        WHERE settlement_point=$1 AND delivery_date=$2 GROUP BY 1)
            SELECT da.hour_ending, abs(da.price_usd_per_mwh - rt.p) g FROM da_spp_hourly da JOIN rt USING (hour_ending)
            WHERE da.settlement_point=$1 AND da.delivery_date=$2 ORDER BY g DESC, da.hour_ending LIMIT 2""",
                               [p["sp"], as_date(p["date"])]).fetchall()
        if len(rows) < 2 or abs(rows[0][1] - rows[1][1]) < 0.01:
            return None
        return int(rows[0][0])


class HubSpreadMaxDay(Template):
    id = "hub_spread_max_day"
    family = FAMILY
    difficulty = 2
    answer_type = "category"
    requires = ("rt_spp_15min",)
    PAIRS = [("HB_WEST", "HB_NORTH"), ("HB_HOUSTON", "HB_NORTH"), ("HB_SOUTH", "HB_HOUSTON"), ("HB_PAN", "HB_WEST")]

    def sample_params(self, rng, ctx):
        months = ctx.months("rt_spp_15min")
        if not months:
            return None
        a, b = self.pick(rng, self.PAIRS)
        return {"a": a, "b": b, "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"During {month_name(p['month'])}, on which operating day was the spread {p['a']} minus {p['b']} "
                f"largest, where each hub's daily price is the average of its 15-minute real-time prices that day? "
                f"Answer with the date as YYYY-MM-DD.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        rows = ctx.con.execute("""
            SELECT delivery_date, avg(price_usd_per_mwh) FILTER (WHERE settlement_point=$1)
                                - avg(price_usd_per_mwh) FILTER (WHERE settlement_point=$2) s
            FROM rt_spp_15min WHERE delivery_date BETWEEN $3 AND $4 GROUP BY 1 ORDER BY s DESC LIMIT 2""",
                               [p["a"], p["b"], a, b]).fetchall()
        if len(rows) < 2 or abs(rows[0][1] - rows[1][1]) < 0.01:
            return None
        return str(rows[0][0])


TEMPLATES = [DaRtSpreadDay(), HubLzBasisMonth(), HighestAvgLoadZoneMonth(), MaxHourlyDaRtGapHe(), HubSpreadMaxDay()]
