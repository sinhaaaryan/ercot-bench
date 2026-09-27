# Data contract (all modules must conform)

Times: store UTC as tz-aware `datetime64[ns, UTC]`. Operating day / hour-ending are ERCOT
Central Prevailing Time conventions (HE 1..24; DST days have 23/25 hours).
CUTOFF(D) = 10:00 America/Chicago on D-1 (day before operating day D), as UTC.

Join key for hourly tables: (operating_date, hour_ending, dst_repeated_hour, zone). On the fall-back
DST day HE2 appears twice; the second has dst_repeated_hour=True (ERCOT DSTFlag).

Zone keys (lowercase, match /hackathon/ercot-bench load_hourly columns without `_mw`):
coast, east, far_west, north, north_central, south, south_central, west, system_total
(ERCOT "Southern" == `south`, "SouthCentral" == `south_central`, "NorthCentral" == `north_central`,
"FarWest" == `far_west`).

## data/parquet/lf_vintage.parquet  (ERCOT NP3-565-CD, load forecast by model & weather zone)
operating_date (date), hour_ending (int), zone (str), model (str), in_use (bool),
forecast_mw (float), publish_time_utc (ts UTC), vintage (str: "dam" = latest publish strictly
before CUTOFF(D); "prev" = latest publish strictly before CUTOFF(D) - 24h, for revisions).

## data/parquet/eia930_df.parquet  (FALLBACK target if no ERCOT API keys)
operating_date, hour_ending, zone="system_total", model="EIA930_DF", in_use=True, forecast_mw,
inferred_publish_time_utc = 14:30 CPT on D-1. VERIFIED: EIA's ERCO forecast equals ERCOT's in-use
SystemTotal from the 14:30 D-1 post (0.3 MW MAE), i.e. AFTER the 10:00 cutoff. It may be used as a
fallback TARGET (error of a later forecast is still unknowable at 10:00) but NEVER as an input feature.

## data/parquet/actual_load.parquet  (labels; from /hackathon/ercot-bench/data/ercot.duckdb load_hourly)
operating_date, hour_ending, zone, actual_mw.

## data/parquet/weather_d2.parquet  (Open-Meteo Previous Runs API, `*_previous_day2`)
Built by `ercot_uncertainty.data.weather`; points/offices in `ercot_uncertainty.zones`.
One row per (operating_date, hour_ending, dst_repeated_hour, point). Columns:
operating_date (date), hour_ending (int16), dst_repeated_hour (bool), zone (str), point (str, key in
`zones.POINTS`), valid_time_utc (ts UTC), temp_c, dewpoint_c, cloud_cover_pct, wind_speed_kmh,
precip_mm (float32: mean over the models with data), n_models (int8: models with temp_c),
issued_utc_max (ts UTC: conservative upper bound on when every contributing run was available),
and per-model columns `<col>__<model>` for model in gfs (gfs_global), ecmwf (ecmwf_ifs025),
icon (icon_global) for cross-model spread. `data/parquet/weather_d2_gem.parquet` is the same
schema with a 4th model, gem (gem_global), added; its base columns are 4-model means. It has 38
extra rows on 2024-01-20 where only GEM precip exists (n_models=0, temp_c NaN).
- Hour mapping: Open-Meteo values are instants at valid_time (precip = sum over the preceding hour);
  the ERCOT interval is the hour ENDING at valid_time (HE1 of D ends 01:00 local on D, HE24 ends
  00:00 local on D+1). DST days have 23/25 rows per point (same convention as ercot_time).
- Coverage: GFS temp_c from before 2023-01-01; GFS other vars and ICON/GEM from 2024-01-20;
  ECMWF from 2024-02-04 (dewpoint 2024-03-01). 2023 rows therefore carry GFS temperature only
  (n_models=1, other columns NaN). Rows with no data at all are dropped.
- Zone values: `weather.zone_hourly(df)` = load-weighted (`Point.weight`) mean of point rows.
- Leakage: `previous_day2` at valid time t = latest run initialised <= t - 48h (verified against
  the Single Runs API). For operating day D the newest possible run is 06Z on D-1 (00Z for GEM),
  available on Open-Meteo by ~13:30Z, before CUTOFF(D) (15Z CDT / 16Z CST).
  `weather.assert_issued_before_cutoff(df)` checks `issued_utc_max < CUTOFF(operating_date)`.
  Day-1 lead is NOT used: for hours after ~HE5 it needs the 12Z D-1 run, issued after the cutoff.

## data/parquet/afd.parquet  (NWS Area Forecast Discussions, IEM archive)
office, issued_utc, wmo_header, issue_line, raw_text, reasoning_text, sections_json.

## data/examples/{train,val,test}.jsonl  (one line per (zone, operating_date))
{"id": "2024-07-15_north_central", "zone": ..., "operating_date": "YYYY-MM-DD", "split": ...,
 "hours": [16,17,18,19,20],
 "prompt": [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
 "baseline": {"16": {"p10":..,"p50":..,"p90":..}, ...},     # numeric baseline percentiles (MW)
 "actual_error": {"16": -512.3, ...},                          # actual - ERCOT dam forecast (MW)
 "meta": {...}}                                                 # spread, in_use model, etc.
Reward = ercot_uncertainty.task.reward_percentiles(completion, baseline, actual_error)
(keys converted to int hour-ending).
