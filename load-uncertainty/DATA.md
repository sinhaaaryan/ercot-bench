# Data: what's versioned, and how to rebuild everything else

Everything comes from public sources. Only the small, eval-critical files are in git (see `.gitignore`):

| Versioned file | What it is |
|---|---|
| `data/examples/test.jsonl` | Test set: 1,593 zone-days (2026-03-08 .. 2026-08-31), prompts + labels + LightGBM v2 anchor |
| `data/examples/test_sample400.jsonl` (+ `numeric/`, `text/`, `textswap/`, `textswap_text/`) | Fixed 400-example subset and its prompt variants |
| `data/examples/val_select.jsonl`, `val_winter.jsonl`, `val_sample200.jsonl` | RL checkpoint-selection set, fall/winter holdout, diversity-check sample |
| `data/examples/reward_scale.json` | Per-zone RL reward scale (median train LightGBM v2 loss, MW) |
| `data/parquet/hourly_features_ercot.parquet` | Every zone x day x peak hour: all features, labels, LightGBM v1/v2 and historical percentiles |
| `data/parquet/lf_vintage.parquet` | ERCOT forecast-by-model history (the 09:30 pre-cutoff post + prior-day post), 2023-01 .. 2026-08 |
| `data/parquet/actual_load.parquet`, `weather_d2.parquet` | Labels; day-2 weather forecasts |

## Rebuild from scratch

```bash
uv sync
# 1. ERCOT load forecasts by model (needs free ERCOT public API credentials in .env:
#    ERCOT_API_USERNAME, ERCOT_API_PASSWORD, ERCOT_PUBLIC_API_SUBSCRIPTION_KEY; ~5 min)
uv run python -m ercot_uncertainty.data.ercot_lf --start 2023-01-01 --end 2026-08-31
# 2. Actual load by weather zone (labels)
uv run python -m ercot_uncertainty.data.actuals
# 3. NWS Area Forecast Discussions, 9 offices (Iowa Environmental Mesonet archive, no key; ~15 min)
uv run python -m ercot_uncertainty.data.afd --start 2023-01-01 --end 2026-09-25
# 4. Day-2 weather forecasts (Open-Meteo Previous Runs API, no key; ~40 min, rate-limited)
uv run python -m ercot_uncertainty.data.weather --start 2023-01-01 --end 2026-08-31
# 5. Cutoff-aware joins, leakage assertions, LightGBM baselines, prompts, splits
uv run python -m ercot_uncertainty.features.build --source ercot
uv run python scripts/add_reward_scale.py
```

`actuals.py` reads ERCOT's settled native load by weather zone. In this project it came from a teammate's DuckDB
built from ERCOT's annual Native Load workbooks (`gridstatus.get_hourly_load_post_settlements`); point
`actuals.py` at your own copy. Day-ahead and real-time settlement-point prices (for the dollar analyses) come from
ERCOT's annual DAM/RTM SPP archives (NP4-180-ER, NP6-785-ER).

## Timing rules (why nothing leaks)

- Cutoff for operating day D = 10:00 America/Chicago on D−1 (the day-ahead market deadline).
- ERCOT forecast: the latest post strictly before the cutoff (always the 09:30 post).
- NWS discussion: the latest product issued at least 5 minutes before the cutoff.
- Weather: Open-Meteo `previous_day2` values; the newest contributing run is available by about 13:30Z on D−1.
- Recent errors use actuals through D−2 only: ERCOT posts zonal actual load once a day (about 05:50 CT, previous day).
- `features/build.py` asserts every input timestamp < cutoff and fails loudly otherwise; `tests/test_leakage.py`
  re-checks the built files.
