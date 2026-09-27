# Pre-RL sanity review (independent, 2026-09-26)

Scope: data -> features -> examples -> reward -> env -> prime-rl configs -> settlement sim. Everything below was
recomputed from `data/parquet/*` and raw files (ERCOT listings and zips, ERCOT MIS, the teammate's duckdb, read-only)
by scratch scripts. No repo code, data or configs were changed. Nothing ran on the GPU.
Scripts: `/tmp/claude-1000/-hackathon-new/a673a139-4d91-4473-9cc3-aa953f6f8786/scratchpad/review/*.py`.

## Findings

### BLOCKER

**B1. The "Yesterday morning (HE1-8)" error (`err_d1_morning`) uses data that isn't published until after the cutoff.**
- Where: `features/build.py:136-145` (the comment there says "ERCOT posts zonal actual load hourly"); the prompt line is
  at `build.py:358-360`, and the value is a GBM feature (`FEATURES`, `build.py:424`).
- Evidence: ERCOT's only public weather-zone actual load report is NP6-345-CD (MIS reportTypeId 13101). It posts **once a day at
  05:50 CPT** and covers the whole previous operating day. I checked the live MIS list: every post is `...T05:50:00-05:00`.
  I also downloaded the 2026-09-26 post, which contains `09/25/2026` HE1-24. So D-1 HE1-8 zonal actuals first appear at
  05:50 on D, about 20 h after CUTOFF(D). The labels come from the settled native-load workbook, which is published even later.
- Extent: the line appears in 8757/8766 train, 1566/1566 val and 1593/1593 test prompts. It is in the full and numeric variants
  and in all 64 `smoke_real/train` prompts. The text variant doesn't have it.
- Measured impact is small but real. Correlation with the peak-hour error by zone is 0.06-0.56 (west 0.56, north 0.37),
  about the same as or slightly above `err_lag2`. Refitting the GBM without the feature moves val pinball from 117.1 to 117.6 MW
  and test from 133.3 to 133.5 MW. Even so, it breaks the project's hard rule and the system prompt's claim
  ("Everything you are given was available at 10:00"), and a judge who knows ERCOT's reports would spot it.
- Fix:
  1. Delete `add_d1_morning_error`, the prompt line, and the entry in `FEATURES`.
  2. Rebuild `features.build` and then `make_sft_examples.py`.
  3. Rebuild `smoke_real` before the real RL run. The smoke-real run in progress uses the leaky prompts, which is fine for
     plumbing but not for results.

  If the team wants a same-day signal, the only defensible one is system-wide (ERCOT publishes real-time system demand).
  Even that would need the real-time source, not the settled numbers.

### SHOULD-FIX

**S1. The settlement sim (`sim/settlement.py`) flatters "hold back for real time".** The accounting itself is sound (see "Checked and OK").
- (a) RT discharge in interval t is triggered by `rt_price[t]` itself (`settlement.py:101`), which is perfect foresight inside
  the interval. On LZ_HOUSTON 2024 (365 days), the gap total(f=0.2) - total(f=0.8) is -$127k with this trigger and -$150k
  with a one-interval-lagged trigger. The foresight is worth about $23k per year per 20 MW fleet, and it all goes to holding back.
- (b) There is no profitability gate. The policy commits and dispatches even when price < night charge price / RTE +
  $10 degradation.
  - Example: LZ_HOUSTON 2026-07-15. Every f loses money, from -$246 at f=0 to -$503 at f=1.
  - Example: LZ_WEST 2025-11-02. f=1 loses $310.
  - Committing less loses less, so on calm cheap days the uncertainty-aware rule looks better for the wrong reason.
- (c) DA foresight (top-2 hours, threshold = max DA price) is already acknowledged. Leave it, but say it on the slide.
- (d) `load_prices` accepts fall-back days: after dropping the repeated hour it sees 24/96 rows. The docstring says DST days are skipped.
- Fix:
  1. Use a lagged or causal RT trigger (`rt[t-1]`, or a forecast).
  2. Only commit or discharge when price > charge_price/RTE + degradation.
  3. Report the $ uplift both ways.

**S2. `eval.py` defaults can hit the teammate's server and produce a biased sample.**
- `--base-url` defaults to `http://localhost:8000/v1` (`eval.py:258`). That is the teammate's port. Use `:8300`.
- `--limit N` takes the first N rows (`eval.py:191`), and the files are zone-major. The docstring example
  `--split test --limit 200` therefore scores 177 coast days + 23 east days. The same applies to the taskset `limit`
  (`taskset.py:301`). Use `test_sample400` / `val_sample200` (they are verified identical subsets, balanced across zones), or shuffle.

**S3. `max_completion_tokens = 256` is tight for the untrained or base-model leaderboard rows.**
- Answer sizes (Qwen3-4B-2507 tokenizer, worst example):
  - Compact JSON as the SFT model emits it, including the `<think></think>` wrapper: <= 168 tokens.
  - `json.dumps` with spaces: 204 tokens.
  - indent=2: 249 tokens.
  - Fenced + indent=2: 253 tokens.
- Any preamble from the base model gets truncated and scored -1. RL after SFT is fine.
- Fix: for the "untrained 4B" and frontier rows use >= 512, as `eval.py` already does. `TRAINING.md` section 4 uses 256.

**S4. AFD text still reveals the calendar date in a few percent of prompts.** Scan of all 11,925 full prompts:
- Named storms ("Tropical Storm Harold", "Hurricane Lidia"): 3.0%.
- Month+day ("today (July 3rd)", "since July 8th"): 2.1%.
- "4th of July": 0.4%.
- Record lists: stripping 2010-2029 leaves e.g. `(2005,)`.

This hardly matters for Qwen on 2025-09..2026-08 val/test. It can matter for the frontier comparison (a model that remembers the
event can recall outcomes). Fix: mask named-storm names and month-day patterns, or at least report frontier results with and
without those examples.

### NIT

- `configs/prime-rl/rl_smoke_real_1gpu.toml`: the header is copy-pasted. It says "Uses the SYNTHETIC smoke split", and its
  launch commands point at `rl_smoke_1gpu.toml` with `--run.name ercot-unc-smoke`. The body is correct (`smoke_real/*`, which exists).
- `seq_len = max_model_len = 4608`, but the longest prompt is **2,996 tokens** (train `2023-06-16_system_total`; val max 2,755,
  test 2,655; nothing over 4,352). The comments say "<= ~4.3k". 3,328 would roughly double the KV tokens available per group.
- SFT targets are rounded twice: the baseline is rounded to 0.1 and then `round()` is applied. 6,471/131,490 train target numbers
  (~5%) differ by 1 MW from the integers shown in the prompt. Harmless, but the copy isn't exact.
- 4CP line: on days 1-~4 of a month the reference is the previous month's max, but the text still says "so far this month".
- `assert_no_leakage` only checks `fc_publish_utc`, the max over in-use rows. `fc_min`, `fc_max` and `fc_median` use every model. I
  verified all 4.38M `lf_vintage` rows are pre-cutoff, but the guard should check all `dam`/`prev` rows.
- `zones.py` docstring says the primary office = the office of the heaviest point. For `north` that is Wichita Falls (OUN), but the
  primary is FWD. OUN isn't downloaded. The choice is sensible, but the doc is inconsistent.
- The GBM OOF embargo (7 d) is shorter than the feature windows (`err_mae30`, 90-d `hist_*`). Fit rows' features therefore contain
  held-out labels. This is second-order (not target leakage), and val/test are clean because they are fit on train only.
- `system_total` weather weights use mean actual load over the whole period, including test. The weights are static; the effect is negligible.
- Reward: the `MIN_BASELINE_LOSS_MW = 5` floor never binds (minimum baseline loss is 6.8 MW). 10% of val and 15% of train examples
  have a baseline loss < 20 MW (small zones), where a 10 MW shift moves the reward by ~0.5. Consider a 20-25 MW floor to cut reward
  noise. Clipping at -1 also equals `INVALID_REWARD`, so a terrible valid answer scores the same as garbage.
- Persistent zone biases (actual - forecast at peak): north -239 MW (~-14% of load, every year), coast +264, system -624.
  They are learnable, but probably a zone-definition difference between ERCOT's LF zones and the native-load workbook zones.
  Worth one sentence in the demo so nobody asks "is north mislabeled?". I found no sign of a zone swap: magnitudes are
  consistent per zone, and the teammate's system total matches ours.
- Ports 8300/8310/5575 are currently held by our own `ercot-unc-smoke-real` run (PIDs 342377/342433/356350). A second run needs
  different ports or has to wait.

## Checked and OK

- **Forecast vintages.** All 4,377,114 `lf_vintage` rows (every model, both vintages) have publish < cutoff (`prev` < cutoff-24h).
  The age is exactly 0.5 h, i.e. the 09:30 CPT post. The inner CSV names (`...20221231.093000...`) confirm the local post time.
  My own selection from the raw listing JSON (latest post strictly before the target) matches the chosen `doc_id` on every
  (date, vintage), with 0 mismatches.
- **Labels.** Exactly one in-use model per (date, HE, zone). The in-use model is identical across zones for every (date, HE), and
  for 1,335/1,339 days across hours. In-use zones sum to the system total exactly. All 59,625 `actual_error` values equal
  settled actual - in-use dam forecast (max |diff| 0.05 MW, from JSON rounding). The teammate's `load_forecast_hourly`
  (09:30 post, system in-use) matches our `lf_vintage_mis_recent` to 0.0 MW on all 6 overlapping days.
- **Prompt numbers, recomputed from raw parquet** for 60 random examples (20 per split, including month boundaries) plus all
  189 examples on or 1-2 days after a DST change. Every number matches:
  - ERCOT forecast, model min/max, in-use vs median, revision (dam - prev in-use).
  - Temperature, dew point, cloud, and model temperature spread (load-weighted points, ddof=1).
  - Stat p10/p50/p90 (equal to the `baseline` field).
  - `err_lag2` (= D-2), 7-day mean/MAE and 30-day MAE over days <= D-2.
  - The 4CP ratio: actuals <= D-2.
  - AFD office and age.

  The weather `issued_utc_max` is < cutoff on all 600,761 rows (minimum margin 1.0 h, at HE24). The HE mapping of weather and
  actual load is correct on DST days. The AFD is the latest issue before cutoff-5 min in every case. Its WMO time matches the
  issue line; the closest is 6 min before cutoff, and none is older than 10.4 h. 9/11,925 examples have no AFD.
- **Baselines.**
  - The reward baseline is the GBM everywhere; the val choice was GBM (117.1 MW) vs hist (141.7).
  - Train predictions are genuinely OOF: 133.8 MW vs 122.9 in-sample.
  - Val/test predictions equal a fresh fit on train only.
  - `hist_*` recomputed independently: 0/299 mismatches, causal (<= D-2).
  - No feature has |corr| > 0.6 with the target. p10 <= p50 <= p90 in 100% of rows.
  - 80% coverage: train 0.776, val 0.792, test 0.784. By zone it ranges 0.59 (west, val) to 0.91 (coast, val), which is room for the LLM to add value.
- **Reward/env.**
  - The env's `pinball_skill` equals `task.reward_percentiles` on all 1,566 val and 1,500 sampled train examples, with 3 completions each.
  - Echoing the baseline scores exactly 0.
  - Copying the prompt's integers scores |r| <= 0.006.
  - `<think>\n\n</think>\n\n{json}` parses and scores identically to plain JSON.
  - 86 project tests and 7 env tests pass (CPU).
- **Splits.**
  - Train is 2023-01-01..2025-08-31 (974 days), val 2025-09-08..2026-02-28 (174), test 2026-03-08..2026-08-31 (177).
  - There is a 7-day embargo between splits, and no date appears in more than one split.
  - Every split has 9 zones with equal counts: 8,766 / 1,566 / 1,593 examples.
  - `numeric/` and `text/` have the same ids, labels and baselines.
  - `val_sample200`, `test_sample400` and `smoke_real/{train,val}` are exact, zone-balanced subsets.
  - `sft/{train,validation}` is in sync with the current train/val (same ids and prompts).
- **Configs.**
  - All referenced splits exist: `smoke/train|val`, `smoke_real/train|val`, `train`, `val`, `sft/{train,validation}.jsonl`.
  - Model names match between the inference and rl configs (1.7B/1.7B, 4B/4B, and the SFT override is documented for both).
  - `max_model_len == seq_len`, and `max_lora_rank` (32) equals the LoRA rank.
  - `enable_thinking`: true for 4B-Instruct (its template's generation prompt is `<|im_start|>assistant\n`) and false for hybrid 1.7B
    (which prefills an empty think block).
  - Our ports (8300, 8310, 13355, 5575) don't collide with the teammate's (8000, 8001, 8003, 5555, 13345).
- **Settlement accounting.** On 9 real days x 2 zones x 5 values of f, total == DA + RT - charge - degradation, discharged <= 39.2 MWh
  (energy), per-interval output <= 20 MW (power), and doing nothing gives $0.
- **Units and sign.** Prompt, system prompt, labels and reward all use actual - forecast in MW, with positive = load above forecast.
  Zone labels match ERCOT names (Southern = `south`, etc.).

Not verifiable offline: whether Open-Meteo `previous_day2` semantics hold for 2023 GFS history (the builder checked recent
weeks against the Single Runs API), and the in-flight RL itself.
