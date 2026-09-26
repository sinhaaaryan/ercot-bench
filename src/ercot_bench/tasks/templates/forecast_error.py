"""Family: forecast_error -- wind/solar/load forecast error statistics and price behavior.

Note: without ERCOT API keys the forecast tables only cover the ~8-day public MIS window, so these
templates yield few (recent-date) tasks. Set ERCOT_API_* keys and re-ingest for full history.
"""

from __future__ import annotations

import math

from ercot_bench.tasks.templates.base import Template, as_date

FAMILY = "forecast_error"


class ForecastMaeDay(Template):
    id = "renewable_fc_mae_day"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = {"abs": 0.1, "rel": 0.0}
    requires = ("wind_forecast_hourly", "solar_forecast_hourly")

    def sample_params(self, rng, ctx):
        kind = rng.choice(["wind", "solar"])
        days = ctx.days(f"{kind}_forecast_hourly")
        if not days:
            return None
        return {"kind": kind, "date": str(self.pick(rng, days))}

    def render_question(self, p):
        name = "STWPF" if p["kind"] == "wind" else "STPPF"
        return (f"What was the mean absolute error of the day-ahead system-wide {p['kind']} forecast ({name}) versus "
                f"actual {p['kind']} generation over the hours of operating day {p['date']}? Answer in MW, rounded to "
                f"1 decimal.")

    def reference_answer(self, p, ctx):
        v = ctx.scalar(f"SELECT avg(abs(forecast_mw - actual_mw)) FROM {p['kind']}_forecast_hourly WHERE delivery_date=?",
                       [as_date(p["date"])])
        return None if v is None else round(v, 1)


class ForecastBiasDay(Template):
    id = "renewable_fc_bias_day"
    family = FAMILY
    difficulty = 2
    answer_type = "number"
    tolerance = {"abs": 0.1, "rel": 0.0}
    requires = ("wind_forecast_hourly", "solar_forecast_hourly")

    def sample_params(self, rng, ctx):
        kind = rng.choice(["wind", "solar"])
        days = ctx.days(f"{kind}_forecast_hourly")
        if not days:
            return None
        return {"kind": kind, "date": str(self.pick(rng, days))}

    def render_question(self, p):
        return (f"On operating day {p['date']}, what was the average bias of the day-ahead system-wide {p['kind']} "
                f"forecast, defined as the mean over the day's hours of (forecast minus actual generation)? "
                f"Positive means over-forecast. Answer in MW, rounded to 1 decimal.")

    def reference_answer(self, p, ctx):
        v = ctx.scalar(f"SELECT avg(forecast_mw - actual_mw) FROM {p['kind']}_forecast_hourly WHERE delivery_date=?",
                       [as_date(p["date"])])
        return None if v is None else round(v, 1)


class WindOverforecastPeakHe(Template):
    id = "wind_overforecast_peak_he"
    family = FAMILY
    difficulty = 2
    answer_type = "integer"
    requires = ("wind_forecast_hourly",)

    def sample_params(self, rng, ctx):
        days = ctx.days("wind_forecast_hourly", exclude_dst=True)
        if not days:
            return None
        return {"date": str(self.pick(rng, days))}

    def render_question(self, p):
        return (f"On operating day {p['date']}, in which hour ending was the day-ahead wind forecast most above actual "
                f"wind generation (largest forecast minus actual)? If tied, give the earliest. Answer with the "
                f"hour-ending number (1-24).")

    def reference_answer(self, p, ctx):
        rows = ctx.con.execute("SELECT hour_ending, forecast_mw - actual_mw e FROM wind_forecast_hourly "
                               "WHERE delivery_date=? ORDER BY e DESC, hour_ending LIMIT 2", [as_date(p["date"])]).fetchall()
        if len(rows) < 2 or abs(rows[0][1] - rows[1][1]) < 0.01:
            return None
        return int(rows[0][0])


class RtAvgWhenWindOverforecast(Template):
    id = "rt_avg_wind_overforecast"
    family = FAMILY
    difficulty = 3
    answer_type = "number"
    tolerance = {"abs": 0.01, "rel": 0.0}
    requires = ("wind_forecast_hourly", "rt_spp_15min")

    def sample_params(self, rng, ctx):
        days = ctx.days(list(self.requires))
        if not days:
            return None
        d = self.pick(rng, days)
        v = ctx.scalar("SELECT quantile_cont(abs(forecast_mw - actual_mw), 0.5) FROM wind_forecast_hourly "
                       "WHERE delivery_date=?", [d])
        if v is None:
            return None
        return {"sp": self.pick(rng, ctx.settlement_points), "date": str(d), "x": int(math.floor(v / 250) * 250)}

    def render_question(self, p):
        return (f"On operating day {p['date']}, consider the hours in which actual wind generation fell short of the "
                f"day-ahead wind forecast by more than {p['x']:,} MW (forecast minus actual > {p['x']:,}). What was the "
                f"average of the 15-minute real-time prices at {p['sp']} during those hours? "
                f"Answer in $/MWh, rounded to 2 decimals.")

    def reference_answer(self, p, ctx):
        v = ctx.scalar("""
            SELECT avg(r.price_usd_per_mwh) FROM rt_spp_15min r JOIN wind_forecast_hourly w
              ON date_trunc('hour', r.interval_start_utc) = w.interval_start_utc
            WHERE r.settlement_point=$sp AND w.delivery_date=$d AND w.forecast_mw - w.actual_mw > $x""",
                       {"sp": p["sp"], "d": as_date(p["date"]), "x": p["x"]})
        return None if v is None else round(v, 2)


class LoadForecastPeakHe(Template):
    id = "load_fc_peak_he"
    family = FAMILY
    difficulty = 1
    answer_type = "integer"
    requires = ("load_forecast_hourly",)

    def sample_params(self, rng, ctx):
        days = ctx.days("load_forecast_hourly", exclude_dst=True)
        if not days:
            return None
        return {"date": str(self.pick(rng, days))}

    def render_question(self, p):
        return (f"According to ERCOT's day-ahead system load forecast, in which hour ending was load expected to peak "
                f"on operating day {p['date']}? Answer with the hour-ending number (1-24).")

    def reference_answer(self, p, ctx):
        rows = ctx.con.execute("SELECT hour_ending, forecast_mw FROM load_forecast_hourly WHERE delivery_date=? "
                               "ORDER BY forecast_mw DESC LIMIT 2", [as_date(p["date"])]).fetchall()
        if len(rows) < 2 or abs(rows[0][1] - rows[1][1]) < 0.01:
            return None
        return int(rows[0][0])


TEMPLATES = [ForecastMaeDay(), ForecastBiasDay(), WindOverforecastPeakHe(), RtAvgWhenWindOverforecast(), LoadForecastPeakHe()]
