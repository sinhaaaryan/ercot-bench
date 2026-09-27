"""Family: time_conventions -- hour ending vs interval start, local vs UTC, DST days, 15-min vs hourly."""

from __future__ import annotations

import datetime as dt

from ercot_bench.tasks.templates.base import Template, as_date, hours_in_day, r2

FAMILY = "time_conventions"
TOL2 = {"abs": 0.01, "rel": 0.0}


class DaPriceHourEnding(Template):
    id = "da_price_he"
    family = FAMILY
    difficulty = 1
    answer_type = "number"
    tolerance = TOL2
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        days = ctx.days("da_spp_hourly", exclude_dst=True)
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days)), "he": rng.randint(1, 24)}

    def render_question(self, p):
        return (f"What was the day-ahead settlement point price at {p['sp']} for hour ending {p['he']} "
                f"(HE{p['he']:02d}) on operating day {p['date']}? Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        return r2(ctx.scalar("SELECT price_usd_per_mwh FROM da_spp_hourly WHERE settlement_point=? AND delivery_date=? "
                             "AND hour_ending=?", [p["sp"], as_date(p["date"]), p["he"]]))


class RtPriceUtcInterval(Template):
    id = "rt_price_utc_interval"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = TOL2
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        days = ctx.days("rt_spp_15min")
        if not days:
            return None
        d = self.pick(rng, days)
        # a UTC start somewhere in that operating day; pick hours where UTC date != local date half the time
        start = dt.datetime(d.year, d.month, d.day) + dt.timedelta(hours=rng.randint(5, 28), minutes=15 * rng.randint(0, 3))
        return {"sp": self.pick(rng, ctx.settlement_points), "utc_start": start.strftime("%Y-%m-%d %H:%M")}

    def render_question(self, p):
        return (f"What was the real-time settlement point price at {p['sp']} for the 15-minute interval that starts at "
                f"{p['utc_start']} UTC? Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        return r2(ctx.scalar("SELECT price_usd_per_mwh FROM rt_spp_15min WHERE settlement_point=? "
                             "AND interval_start_utc = CAST(? AS TIMESTAMP)", [p["sp"], p["utc_start"]]))


class RtHourAvgHourEnding(Template):
    id = "rt_hour_avg_he"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = TOL2
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        days = ctx.days("rt_spp_15min", exclude_dst=True)
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days)), "he": rng.randint(1, 24)}

    def render_question(self, p):
        return (f"What was the hourly real-time price at {p['sp']} for hour ending {p['he']} on operating day {p['date']}, "
                f"computed as the simple average of the four 15-minute settlement point prices in that hour? "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        return r2(ctx.scalar("SELECT avg(price_usd_per_mwh) FROM rt_spp_15min WHERE settlement_point=? "
                             "AND delivery_date=? AND hour_ending=?", [p["sp"], as_date(p["date"]), p["he"]]))


class DstRepeatedHourDa(Template):
    id = "dst_repeated_hour_da"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = TOL2
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        days = ctx.dst_days("da_spp_hourly", "fall")
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days)),
                "which": rng.choice(["first", "second"])}

    def render_question(self, p):
        tz = "Central Daylight Time" if p["which"] == "first" else "Central Standard Time"
        return (f"{p['date']} was the ERCOT fall-back (end of daylight saving time) day, so the hour from 01:00 to 02:00 "
                f"local time occurred twice. What was the day-ahead settlement point price at {p['sp']} for the "
                f"{p['which']} occurrence of that hour (the one in {tz})? Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        repeated = p["which"] == "second"
        return r2(ctx.scalar("SELECT price_usd_per_mwh FROM da_spp_hourly WHERE settlement_point=? AND delivery_date=? "
                             "AND hour_ending=2 AND dst_repeated_hour=?", [p["sp"], as_date(p["date"]), repeated]))


class IntervalCountDay(Template):
    id = "interval_count_day"
    family = FAMILY
    difficulty = 2
    answer_type = "integer"
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        dst = ctx.dst_days("rt_spp_15min")
        days = ctx.days("rt_spp_15min")
        if not days:
            return None
        d = self.pick(rng, dst) if dst and rng.random() < 0.7 else self.pick(rng, days)
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(d)}

    def render_question(self, p):
        return (f"How many 15-minute real-time settlement intervals does operating day {p['date']} contain for "
                f"{p['sp']}? Answer with an integer.")

    def reference_answer(self, p, ctx):
        return ctx.scalar("SELECT count(*) FROM rt_spp_15min WHERE settlement_point=? AND delivery_date=?",
                          [p["sp"], as_date(p["date"])])


class PeakRtUtc(Template):
    id = "peak_rt_utc"
    family = FAMILY
    difficulty = 2
    answer_type = "timestamp"
    answer_timezone = "UTC"
    requires = ("rt_spp_15min",)

    def sample_params(self, rng, ctx):
        days = ctx.days("rt_spp_15min")
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days))}

    def render_question(self, p):
        return (f"On operating day {p['date']}, which 15-minute interval had the highest real-time settlement point "
                f"price at {p['sp']}? If tied, take the earliest. Give the interval start time in UTC "
                f"(format YYYY-MM-DD HH:MM).")

    def reference_answer(self, p, ctx):
        v = ctx.scalar("SELECT interval_start_utc FROM rt_spp_15min WHERE settlement_point=? AND delivery_date=? "
                       "ORDER BY price_usd_per_mwh DESC, interval_start_utc ASC LIMIT 1", [p["sp"], as_date(p["date"])])
        return None if v is None else v.strftime("%Y-%m-%d %H:%M")


class DaDailyAvgDstDay(Template):
    id = "da_daily_avg_dst_day"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = TOL2
    requires = ("da_spp_hourly",)

    def sample_params(self, rng, ctx):
        days = ctx.dst_days("da_spp_hourly")
        if not days:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(self.pick(rng, days))}

    def render_question(self, p):
        return (f"What was the average day-ahead settlement point price at {p['sp']} over all delivery hours of "
                f"operating day {p['date']} (a daylight-saving transition day)? Weight each delivered hour equally. "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        d = as_date(p["date"])
        n = ctx.scalar("SELECT count(*) FROM da_spp_hourly WHERE settlement_point=? AND delivery_date=?", [p["sp"], d])
        if n != hours_in_day(d):
            return None
        return r2(ctx.scalar("SELECT avg(price_usd_per_mwh) FROM da_spp_hourly WHERE settlement_point=? "
                             "AND delivery_date=?", [p["sp"], d]))


TEMPLATES = [DaPriceHourEnding(), RtPriceUtcInterval(), RtHourAvgHourEnding(), DstRepeatedHourDa(),
             IntervalCountDay(), PeakRtUtc(), DaDailyAvgDstDay()]
