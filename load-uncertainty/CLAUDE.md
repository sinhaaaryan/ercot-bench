# ERCOT forecast-uncertainty predictor (hackathon, Track 1)

Predict the distribution of ERCOT's day-ahead load-forecast error (actual - forecast) per weather zone,
peak hours, from numeric signals + NWS Area Forecast Discussion text. Full brief: docs/BRIEF.md.

## Hard rules
- Cutoff = 10:00 America/Chicago, day before operating day. Every input timestamp must be < cutoff;
  features/ asserts this and must fail loudly.
- All task logic (prompt, parse, reward) lives in plain Python in src/ercot_uncertainty/task.py;
  the Verifiers env only imports it.
- Keep data from before and after 2025-12-05 (RTC+B) separate for price-based analysis.
- Shell commands given to the user: give BOTH a one-line and a multi-line version.

## Shared machine (teammate "aaryan" works in /hackathon)
- GPU is shared. Check `/usr/lib/wsl/lib/nvidia-smi` before launching anything on the GPU;
  never kill or pause the teammate's processes. A mem_guard in /hackathon pauses jobs near 31.8GB.
- /hackathon/prime-rl/.venv has prime-rl + verifiers + vLLM 0.29 (torch 2.13 cu130). Use it read-only.
- /hackathon/ercot-bench/data/ercot.duckdb has prices, actual load by weather zone, fuel mix
  (read-only reuse is fine; do not write there).
- Frontier eval uses the Claude Code CLI (`claude -p`), no API keys.

## Data sources verified
- NWS AFDs: IEM `https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=AFDFWD&sdate=...&edate=...&fmt=text`
- Open-Meteo historical-forecast API goes back to 2023 but stitches the earliest hours of each run (too close to
  a nowcast, so it leaks); use the Previous Runs API (`*_previous_day1`) for day-ahead weather. The
  ensemble API only keeps about 4 months, so there is no history of ensemble spread.
- ERCOT load-forecast vintages by model: public MIS keeps ~8 days; history requires ERCOT public API
  credentials (ERCOT_API_USERNAME, ERCOT_API_PASSWORD, ERCOT_PUBLIC_API_SUBSCRIPTION_KEY in .env).
