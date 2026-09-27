"""Data-quality report: rows per month, gaps, duplicates, DST transition days, null rates."""

from __future__ import annotations

from pathlib import Path

from ercot_bench.db import TABLES, connect_readonly

# table -> (group key for per-entity checks, expected minutes per interval)
TABLE_META = {
    "rt_spp_15min": ("settlement_point", 15),
    "da_spp_hourly": ("settlement_point", 60),
    "load_hourly": (None, 60),
    "fuel_mix_15min": ("fuel", 15),
    "as_prices_dam_hourly": ("service", 60),
    "wind_forecast_hourly": (None, 60),
    "solar_forecast_hourly": (None, 60),
    "load_forecast_hourly": (None, 60),
}


def quality_report(db_path: Path | str) -> str:
    con = connect_readonly(db_path)
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    out: list[str] = []
    issues: list[str] = []
    for t in TABLES:
        if t not in tables:
            out.append(f"## {t}: MISSING")
            issues.append(f"{t}: table missing")
            continue
        key, minutes = TABLE_META[t]
        n, lo, hi = con.execute(f"SELECT count(*), min(interval_start_utc), max(interval_start_utc) FROM {t}").fetchone()
        n_ent = con.execute(f"SELECT count(DISTINCT {key}) FROM {t}").fetchone()[0] if key else 1
        out.append(f"## {t}\nrows={n:,} entities={n_ent} utc_range=[{lo} .. {hi}]")

        # rows per month
        rows = con.execute(f"""
            SELECT strftime(delivery_date, '%Y-%m') m, count(*) FROM {t} GROUP BY 1 ORDER BY 1""").fetchall()
        out.append("rows/month: " + " ".join(f"{m}:{c}" for m, c in rows))

        # duplicates on (key, utc start)
        kcols = f"{key}, interval_start_utc" if key else "interval_start_utc"
        dups = con.execute(f"SELECT count(*) FROM (SELECT {kcols} FROM {t} GROUP BY ALL HAVING count(*) > 1)").fetchone()[0]
        out.append(f"duplicate (entity, interval) rows: {dups}")
        if dups:
            issues.append(f"{t}: {dups} duplicate keys")

        # gaps: consecutive UTC starts per entity differing by more than one interval
        part = f"PARTITION BY {key}" if key else ""
        gaps = con.execute(f"""
            WITH x AS (
              SELECT {key + ',' if key else ''} interval_start_utc s,
                     lag(interval_start_utc) OVER ({part} ORDER BY interval_start_utc) p FROM {t})
            SELECT {key + ',' if key else "'all',"} p, s, datediff('minute', p, s) gap_min FROM x
            WHERE p IS NOT NULL AND datediff('minute', p, s) > {minutes}
            ORDER BY gap_min DESC LIMIT 5""").fetchall()
        n_gaps = con.execute(f"""
            WITH x AS (SELECT interval_start_utc s, lag(interval_start_utc) OVER ({part} ORDER BY interval_start_utc) p
                       FROM {t}) SELECT count(*) FROM x WHERE p IS NOT NULL AND datediff('minute', p, s) > {minutes}
        """).fetchone()[0]
        out.append(f"gaps: {n_gaps}" + ("; largest: " + "; ".join(f"{g[0]} {g[1]}->{g[2]} ({g[3]} min)" for g in gaps) if gaps else ""))
        if n_gaps:
            issues.append(f"{t}: {n_gaps} gaps (largest {gaps[0][3]} min)")

        # DST transition days: intervals per entity per day (expect 23h/25h)
        per_hour = 60 // minutes
        dst = con.execute(f"""
            SELECT delivery_date, count(*) / {n_ent} AS per_entity, sum(dst_repeated_hour::INT) / {n_ent} AS repeated
            FROM {t} WHERE (month(delivery_date) = 3 AND day(delivery_date) BETWEEN 8 AND 14 AND dayofweek(delivery_date) = 0)
                        OR (month(delivery_date) = 11 AND day(delivery_date) BETWEEN 1 AND 7 AND dayofweek(delivery_date) = 0)
            GROUP BY 1 ORDER BY 1""").fetchall()
        for d, c, r in dst:
            exp = (23 if d.month == 3 else 25) * per_hour
            flag = "" if c == exp else f"  <-- expected {exp}"
            out.append(f"DST day {d}: {c:g} intervals/entity, repeated-hour intervals {r:g}{flag}")
            if c != exp:
                issues.append(f"{t}: DST day {d} has {c:g} intervals/entity, expected {exp}")

        # null rates
        cols = [r[0] for r in con.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name='{t}'").fetchall()]
        nulls = con.execute("SELECT " + ", ".join(f"avg(({c} IS NULL)::INT)" for c in cols) + f" FROM {t}").fetchone()
        nz = [f"{c}={v:.2%}" for c, v in zip(cols, nulls) if v]
        out.append("null rates: " + (", ".join(nz) if nz else "none"))
        for c, v in zip(cols, nulls):
            if v and v > 0.001:
                issues.append(f"{t}.{c}: {v:.2%} null")
        out.append("")
    con.close()
    out.append("## SUMMARY")
    out.extend([f"- {i}" for i in issues] or ["- clean"])
    return "\n".join(out)
