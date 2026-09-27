"""Family: basic_aggregates -- average/max/min price at a location over a period."""

from __future__ import annotations

from ercot_bench.tasks.templates.base import Template, as_date, month_bounds, month_name, r2

FAMILY = "basic_aggregates"
TOL2 = {"abs": 0.01, "rel": 0.0}


class RtAvgDay(Template):
    id = "rt_avg_day"
    family = FAMILY
    difficulty = 1
    answer_type = "number"
    tolerance = TOL2
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        days = ctx.days("rt_spp_15min")
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days))}

    def render_question(self, p):
        return (f"What was the average real-time settlement point price at {p['sp']} on operating day {p['date']}, "
                f"averaging all of that day's 15-minute intervals? Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        return r2(ctx.scalar("SELECT avg(price_usd_per_mwh) FROM rt_spp_15min WHERE settlement_point=? AND delivery_date=?",
                             [p["sp"], as_date(p["date"])]))


class DaMaxMonth(Template):
    id = "da_max_month"
    family = FAMILY
    difficulty = 1
    answer_type = "number"
    tolerance = TOL2
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        months = ctx.months("da_spp_hourly")
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"What was the highest hourly day-ahead settlement point price at {p['sp']} during {month_name(p['month'])}? "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        return r2(ctx.scalar("SELECT max(price_usd_per_mwh) FROM da_spp_hourly WHERE settlement_point=? "
                             "AND delivery_date BETWEEN ? AND ?", [p["sp"], a, b]))


class RtMinMonth(Template):
    id = "rt_min_month"
    family = FAMILY
    difficulty = 1
    answer_type = "number"
    tolerance = TOL2
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        months = ctx.months("rt_spp_15min")
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"What was the lowest 15-minute real-time settlement point price at {p['sp']} during "
                f"{month_name(p['month'])}? Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        return r2(ctx.scalar("SELECT min(price_usd_per_mwh) FROM rt_spp_15min WHERE settlement_point=? "
                             "AND delivery_date BETWEEN ? AND ?", [p["sp"], a, b]))


class AsAvgMonth(Template):
    id = "as_avg_month"
    family = FAMILY
    difficulty = 1
    answer_type = "number"
    tolerance = TOL2
    requires = ("as_prices_dam_hourly",)
    SERVICES = {"REGUP": "Regulation Up", "REGDN": "Regulation Down", "RRS": "Responsive Reserve Service",
                "NSPIN": "Non-Spinning Reserve", "ECRS": "ERCOT Contingency Reserve Service"}

    def sample_params(self, rng, ctx):
        months = ctx.months("as_prices_dam_hourly")
        if not months:
            return None
        return {"service": self.pick(rng, sorted(self.SERVICES)), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"What was the average hourly day-ahead clearing price for {self.SERVICES[p['service']]} "
                f"({p['service']}) during {month_name(p['month'])}? Answer in $/MW, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        return r2(ctx.scalar("SELECT avg(price_usd_per_mw) FROM as_prices_dam_hourly WHERE service=? "
                             "AND delivery_date BETWEEN ? AND ?", [p["service"], a, b]))


class RtAvgCentsPerKwhMonth(Template):
    id = "rt_avg_cents_kwh_month"
    family = FAMILY
    difficulty = 1
    answer_type = "number"
    tolerance = {"abs": 0.001, "rel": 0.0}
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        months = ctx.months("rt_spp_15min")
        if not months:
            return None
        return {"sp": self.pick(rng, ctx.load_zones), "month": self.pick(rng, months)}

    def render_question(self, p):
        return (f"A residential customer wants the average real-time wholesale price at {p['sp']} during "
                f"{month_name(p['month'])} (average of all 15-minute intervals) expressed in cents per kWh. "
                f"Answer in cents/kWh, rounded to 3 decimals.")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        v = ctx.scalar("SELECT avg(price_usd_per_mwh) / 10 FROM rt_spp_15min WHERE settlement_point=? "
                       "AND delivery_date BETWEEN ? AND ?", [p["sp"], a, b])
        return None if v is None else round(v, 3)


TEMPLATES = [RtAvgDay(), DaMaxMonth(), RtMinMonth(), AsAvgMonth(), RtAvgCentsPerKwhMonth()]
