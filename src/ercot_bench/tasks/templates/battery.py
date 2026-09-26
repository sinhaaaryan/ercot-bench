"""Family: battery -- revenue arithmetic for simple battery schedules."""

from __future__ import annotations

from ercot_bench.tasks.templates.base import Template, as_date, month_bounds, month_name

FAMILY = "battery"
SIZES = [1, 5, 10, 20, 50, 100]
EFFS = [0.80, 0.85, 0.88, 0.90, 1.0]


def _schedule(rng):
    k = rng.choice([1, 2, 3, 4])
    c0 = rng.randint(1, 16 - k)          # charge window starts HE1..HE(16-k)
    d0 = rng.randint(max(c0 + k, 16), 24 - k + 1)  # discharge after charging, in the evening
    return {"c_start": c0, "c_end": c0 + k - 1, "d_start": d0, "d_end": d0 + k - 1}


def _schedule_text(p, market):
    return (f"It charges at {p['mw']} MW during hours ending {p['c_start']} through {p['c_end']} and discharges during "
            f"hours ending {p['d_start']} through {p['d_end']}. Round-trip efficiency is {p['eff']:.0%}, applied on "
            f"discharge: the discharge power is {p['mw']} MW x {p['eff']:.2f}. Revenue is discharge energy times price "
            f"minus charging energy times price, using {market} settlement point prices at {p['sp']}.")


class BatteryFixedDaRevenue(Template):
    id = "battery_fixed_da_revenue"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = {"abs": 0.5, "rel": 1e-4}
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        days = ctx.days("da_spp_hourly", exclude_dst=True)
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days)),
                "mw": rng.choice(SIZES), "eff": rng.choice(EFFS), **_schedule(rng)}

    def render_question(self, p):
        return (f"A battery follows a fixed schedule on operating day {p['date']}. " + _schedule_text(p, "hourly day-ahead")
                + " What is the day's net revenue in dollars, rounded to 2 decimals?")

    def reference_answer(self, p, ctx):
        v = ctx.scalar("""
            SELECT sum(CASE WHEN hour_ending BETWEEN $ds AND $de THEN $mw * $eff * price_usd_per_mwh
                            WHEN hour_ending BETWEEN $cs AND $ce THEN -$mw * price_usd_per_mwh ELSE 0 END)
            FROM da_spp_hourly WHERE settlement_point=$sp AND delivery_date=$d""",
                       {"ds": p["d_start"], "de": p["d_end"], "cs": p["c_start"], "ce": p["c_end"], "mw": p["mw"],
                        "eff": p["eff"], "sp": p["sp"], "d": as_date(p["date"])})
        return None if v is None else round(v, 2)


class BatteryFixedRtRevenue(Template):
    id = "battery_fixed_rt_revenue"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = {"abs": 0.5, "rel": 1e-4}
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        days = ctx.days("rt_spp_15min", exclude_dst=True)
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days)),
                "mw": rng.choice(SIZES), "eff": rng.choice(EFFS), **_schedule(rng)}

    def render_question(self, p):
        return (f"A battery follows a fixed schedule on operating day {p['date']}, settled on 15-minute real-time prices. "
                + _schedule_text(p, "15-minute real-time")
                + " Each 15-minute interval's energy is MW x 0.25 h. What is the day's net revenue in dollars, rounded "
                  "to 2 decimals?")

    def reference_answer(self, p, ctx):
        v = ctx.scalar("""
            SELECT sum(0.25 * CASE WHEN hour_ending BETWEEN $ds AND $de THEN $mw * $eff * price_usd_per_mwh
                                   WHEN hour_ending BETWEEN $cs AND $ce THEN -$mw * price_usd_per_mwh ELSE 0 END)
            FROM rt_spp_15min WHERE settlement_point=$sp AND delivery_date=$d""",
                       {"ds": p["d_start"], "de": p["d_end"], "cs": p["c_start"], "ce": p["c_end"], "mw": p["mw"],
                        "eff": p["eff"], "sp": p["sp"], "d": as_date(p["date"])})
        return None if v is None else round(v, 2)


class BatteryBestArbitrageDa(Template):
    id = "battery_best_arbitrage_da"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = {"abs": 0.5, "rel": 1e-4}
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        days = ctx.days("da_spp_hourly", exclude_dst=True)
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days)),
                "mw": rng.choice(SIZES), "eff": rng.choice(EFFS)}

    def render_question(self, p):
        return (f"A {p['mw']} MW / {p['mw']} MWh battery (1-hour duration) starts empty on operating day {p['date']} and "
                f"may do at most one cycle: charge fully in one hour, then discharge in a strictly later hour of the same "
                f"day. Round-trip efficiency is {p['eff']:.0%} (discharged energy = {p['eff']:.2f} x charged energy). "
                f"Using hourly day-ahead prices at {p['sp']}, what is the maximum achievable net revenue in dollars "
                f"(0 if no profitable cycle exists), rounded to 2 decimals?")

    def reference_answer(self, p, ctx):
        v = ctx.scalar("""
            SELECT greatest(0, max($mw * ($eff * d.price_usd_per_mwh - c.price_usd_per_mwh)))
            FROM da_spp_hourly c JOIN da_spp_hourly d
              ON d.settlement_point = c.settlement_point AND d.delivery_date = c.delivery_date
             AND d.interval_start_utc > c.interval_start_utc
            WHERE c.settlement_point=$sp AND c.delivery_date=$d""",
                       {"mw": p["mw"], "eff": p["eff"], "sp": p["sp"], "d": as_date(p["date"])})
        return None if v is None else round(v, 2)


class BatteryMonthlyFixedDaRevenue(Template):
    id = "battery_monthly_fixed_da_revenue"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = {"abs": 1.0, "rel": 1e-4}
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        months = ctx.months("da_spp_hourly")
        if not months:
            return None
        k = rng.choice([1, 2])
        c0 = rng.randint(10, 15 - k)  # midday charging (never touches the DST hours)
        d0 = rng.randint(18, 22 - k)
        return {"sp": self.pick(rng, ctx.settlement_points), "month": self.pick(rng, months), "mw": rng.choice(SIZES),
                "eff": rng.choice(EFFS), "c_start": c0, "c_end": c0 + k - 1, "d_start": d0, "d_end": d0 + k - 1}

    def render_question(self, p):
        return (f"A battery repeats the same fixed schedule every operating day of {month_name(p['month'])}. "
                + _schedule_text(p, "hourly day-ahead") + " What is the total net revenue for the month in dollars, "
                                                          "rounded to 2 decimals?")

    def reference_answer(self, p, ctx):
        a, b = month_bounds(p["month"])
        v = ctx.scalar("""
            SELECT sum(CASE WHEN hour_ending BETWEEN $ds AND $de THEN $mw * $eff * price_usd_per_mwh
                            WHEN hour_ending BETWEEN $cs AND $ce THEN -$mw * price_usd_per_mwh ELSE 0 END)
            FROM da_spp_hourly WHERE settlement_point=$sp AND delivery_date BETWEEN $a AND $b""",
                       {"ds": p["d_start"], "de": p["d_end"], "cs": p["c_start"], "ce": p["c_end"], "mw": p["mw"],
                        "eff": p["eff"], "sp": p["sp"], "a": a, "b": b})
        return None if v is None else round(v, 2)


TEMPLATES = [BatteryFixedDaRevenue(), BatteryFixedRtRevenue(), BatteryBestArbitrageDa(), BatteryMonthlyFixedDaRevenue()]
