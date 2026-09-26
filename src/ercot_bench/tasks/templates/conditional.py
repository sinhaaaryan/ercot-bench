"""Family: conditional -- price statistics conditioned on another table (load, net load, generation)."""

from __future__ import annotations

import math

from ercot_bench.tasks.templates.base import Template, month_bounds, month_name, r2

FAMILY = "conditional"
TOL2 = {"abs": 0.01, "rel": 0.0}

NET_LOAD_CTE = """
    gen AS (SELECT date_trunc('hour', interval_start_utc) h,
                   avg(generation_mw) FILTER (WHERE fuel = 'Wind') wind,
                   avg(generation_mw) FILTER (WHERE fuel = 'Solar') solar
            FROM fuel_mix_15min WHERE delivery_date BETWEEN $a AND $b GROUP BY 1),
    nl AS (SELECT l.interval_start_utc h, l.ercot_total_mw - g.wind - g.solar AS net_load
           FROM load_hourly l JOIN gen g ON g.h = l.interval_start_utc
           WHERE l.delivery_date BETWEEN $a AND $b)"""


class RtAvgHighNetLoadMonth(Template):
    id = "rt_avg_high_netload_month"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = TOL2
    requires = ("rt_spp_15min", "load_hourly", "fuel_mix_15min")

    def sample_params(self, rng, ctx):
        months = ctx.months(list(self.requires))
        if not months:
            return None
        ym = self.pick(rng, months)
        a, b = month_bounds(ym)
        q = rng.choice([0.75, 0.9])
        v = ctx.scalar(f"WITH {NET_LOAD_CTE} SELECT quantile_cont(net_load, {q}) FROM nl", {"a": a, "b": b})
        if v is None:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": ym, "threshold_mw": int(math.floor(v / 1000) * 1000)}

    def render_question(self, p):
        return (f"During {month_name(p['month'])}, consider the hours in which ERCOT hourly net load exceeded "
                f"{p['threshold_mw']:,} MW, where net load = system load (ercot_total_mw) minus wind generation minus "
                f"solar generation, with hourly wind and solar taken as the average of that hour's 15-minute "
                f"generation. What was the average of all 15-minute real-time prices at {p['sp']} within those hours? "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        return r2(ctx.scalar(f"""
            WITH {NET_LOAD_CTE}
            SELECT avg(r.price_usd_per_mwh) FROM rt_spp_15min r JOIN nl ON date_trunc('hour', r.interval_start_utc) = nl.h
            WHERE r.settlement_point = $sp AND nl.net_load > $x""",
                             {"a": a, "b": b, "sp": p["sp"], "x": p["threshold_mw"]}))


class SolarShareMaxDay(Template):
    id = "solar_share_max_day"
    family = FAMILY
    difficulty = 2
    answer_type = "category"
    requires = ("fuel_mix_15min",)

    def sample_params(self, rng, ctx):
        months = ctx.months("fuel_mix_15min")
        if not months:
            return None
        return {"month": self.pick(rng, months), "fuel": rng.choice(["Solar", "Wind"])}

    def render_question(self, p):
        return (f"During {month_name(p['month'])}, on which operating day was {p['fuel'].lower()}'s share of total "
                f"generation highest? Compute each day's share as {p['fuel'].lower()} energy divided by the total energy "
                f"of all fuel types excluding WSL (storage charging). Answer with the date as YYYY-MM-DD.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        rows = ctx.con.execute("""
            SELECT delivery_date, sum(energy_mwh) FILTER (WHERE fuel = $f) / sum(energy_mwh) FILTER (WHERE fuel <> 'WSL') s
            FROM fuel_mix_15min WHERE delivery_date BETWEEN $a AND $b GROUP BY 1 ORDER BY s DESC LIMIT 2""",
                               {"f": p["fuel"], "a": a, "b": b}).fetchall()
        if len(rows) < 2 or abs(rows[0][1] - rows[1][1]) < 1e-4:
            return None
        return str(rows[0][0])


class WindAvgNegativeRt(Template):
    id = "wind_avg_negative_rt"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = {"abs": 1.0, "rel": 0.0}
    requires = ("rt_spp_15min", "fuel_mix_15min")

    def sample_params(self, rng, ctx):
        months = ctx.months(list(self.requires))
        if not months:
            return None
        return {"sp": self.pick(rng, ["HB_WEST", "HB_PAN", "HB_NORTH", "HB_SOUTH", "LZ_WEST"]), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"During {month_name(p['month'])}, what was the average ERCOT wind generation (MW) across the 15-minute "
                f"intervals in which the real-time price at {p['sp']} was negative (below $0/MWh)? "
                f"Answer in MW, rounded to the nearest whole MW.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        n, v = ctx.con.execute("""
            SELECT count(*), avg(f.generation_mw) FROM rt_spp_15min r
            JOIN fuel_mix_15min f ON f.interval_start_utc = r.interval_start_utc AND f.fuel = 'Wind'
            WHERE r.settlement_point = $sp AND r.delivery_date BETWEEN $a AND $b AND r.price_usd_per_mwh < 0""",
                               {"sp": p["sp"], "a": a, "b": b}).fetchone()
        if not n or n < 4:
            return None
        return float(round(v))


class DaAvgHighLoadHours(Template):
    id = "da_avg_high_load_hours"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = TOL2
    requires = ("da_spp_hourly", "load_hourly")

    def sample_params(self, rng, ctx):
        months = ctx.months(list(self.requires))
        if not months:
            return None
        ym = self.pick(rng, months)
        a, b = month_bounds(ym)
        q = rng.choice([0.5, 0.8, 0.95])
        v = ctx.scalar(f"SELECT quantile_cont(ercot_total_mw, {q}) FROM load_hourly WHERE delivery_date BETWEEN ? AND ?", [a, b])
        if v is None:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": ym, "threshold_mw": int(math.floor(v / 1000) * 1000)}

    def render_question(self, p):
        return (f"What was the average day-ahead settlement point price at {p['sp']} during the hours of "
                f"{month_name(p['month'])} in which actual ERCOT system load exceeded {p['threshold_mw']:,} MW? "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        return r2(ctx.scalar("""
            SELECT avg(d.price_usd_per_mwh) FROM da_spp_hourly d JOIN load_hourly l USING (interval_start_utc)
            WHERE d.settlement_point = $sp AND d.delivery_date BETWEEN $a AND $b AND l.ercot_total_mw > $x""",
                             {"sp": p["sp"], "a": a, "b": b, "x": p["threshold_mw"]}))


class GasGenAtPeakPrice(Template):
    id = "gas_gen_at_peak_price"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = {"abs": 1.0, "rel": 0.0}
    requires = ("rt_spp_15min", "fuel_mix_15min")

    def sample_params(self, rng, ctx):
        months = ctx.months(list(self.requires))
        if not months:
            return None
        return {"sp": self.pick(rng, ["HB_HUBAVG", "HB_BUSAVG", "HB_NORTH", "HB_HOUSTON"]), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"Find the 15-minute interval in {month_name(p['month'])} with the highest real-time price at {p['sp']} "
                f"(earliest if tied). What was total natural-gas generation in that interval, counting both the 'Gas' "
                f"and 'Gas-CC' fuel types? Answer in MW (average MW over the interval), rounded to the nearest whole MW.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        v = ctx.scalar("""
            WITH pk AS (SELECT interval_start_utc FROM rt_spp_15min
                        WHERE settlement_point = $sp AND delivery_date BETWEEN $a AND $b
                        ORDER BY price_usd_per_mwh DESC, interval_start_utc LIMIT 1)
            SELECT sum(generation_mw) FROM fuel_mix_15min f JOIN pk USING (interval_start_utc)
            WHERE f.fuel IN ('Gas', 'Gas-CC')""", {"sp": p["sp"], "a": a, "b": b})
        return None if v is None else float(round(v))


TEMPLATES = [RtAvgHighNetLoadMonth(), SolarShareMaxDay(), WindAvgNegativeRt(), DaAvgHighLoadHours(), GasGenAtPeakPrice()]
