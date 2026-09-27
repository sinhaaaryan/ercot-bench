# ERCOT load-forecast uncertainty eval

## Task
For one ERCOT weather zone (or the system total) and one operating day D, predict the 10th/50th/90th percentiles
of ERCOT's day-ahead load-forecast error (actual - in-use forecast, MW) for hours ending 16-20, using only
information published before 10:00 America/Chicago on D-1 (the day-ahead market deadline).

## Data
- Test: 1,593 zone-days, 2026-03-08 .. 2026-08-31 (177 days x 9 series). Strictly after all training/validation data.
- Fixed subsets for cheaper comparisons: `test_sample400` (400 zone-days, 162 days) and its first 100 rows.
- Prompt variants: full (numbers + forecaster text), `numeric/` (numbers only), `text/` (anchor + text only).
  Every variant shows the reference numeric model's p10/p50/p90 as an anchor.
- Leakage: every input's publish/issue time is asserted < cutoff (`features/build.py`, `tests/test_leakage.py`);
  dates/years are stripped from the forecaster text.

## Scoring (one rule set, `eval_report.py`)
- **Pinball loss** (mean over p10/p50/p90 and hours), in MW. Proper scoring rule: always-wide hedging loses.
- **Zone-normalized skill**: mean over the 8 weather zones of each zone's % improvement vs the reference, equal
  weight, because in MW the system total (34%), North Central (19%) and Coast (17%) dominate. System total is
  reported separately.
- **Calibration**: share of actuals below each percentile (targets 10/50/90%), overall and by zone, plus a sigma
  view (p10-p90 read as +/-1.28 sigma: within 1 sigma ~68%, 2 sigma ~95%, beyond 3 sigma ~0.3%). ERCOT's errors
  are skewed and fat-tailed (beyond 3 sigma about 2.7x the normal rate), which is why the model outputs
  percentiles, not mean +/- sigma.
- **Parse failures** are scored as "trust ERCOT exactly" (p10 = p50 = p90 = 0) and the rate is reported separately.
- **Uncertainty**: all CIs are day-clustered bootstraps (zone-days on one date share weather and text).
  Detectable effect (95% CI half-width): about +/-3.4% on `test_sample400`, +/-2.0% on the full test set.
- **Slices**: ERCOT models agreed/disagreed, big-miss days, discussion themes, zone, month.
- **Dollars** (optional): `sim/settlement.py` battery rule (commit less day-ahead when predicted uncertainty is high)
  vs a fixed commitment, 1,000-unit fleet, price taker.

## Declared headline comparison (fixed before the RL run)
**RL-trained Qwen3-4B vs LightGBM v2 on the FULL test set**, reported as pinball MW and zone-normalized skill with
day-clustered 95% CIs. Secondary: SFT-only model, untrained model, Opus 5.5 (full and text-only prompts), LightGBM v1.
Everything else (100/400 subsets, thinking variants, older anchors in `results/lgbm_v1_anchor/`) is exploratory.

## Validation use (declared before RL)
- `val_select` (2025-09-08 .. 2025-11-30, 756 zone-days): RL checkpoint selection (best greedy pinball), evaluated every
  25 steps. Also used earlier for the LightGBM audit, so it is not a clean test.
- `val_winter` (2025-12-01 .. 2026-02-28, 810 zone-days): no performance-based LLM decision uses it. (SFT's validation
  loss and the pre-RL diversity/format gate drew on all of validation, including these dates, but neither chose between
  models by forecast skill.) Reported as a secondary fall/winter test. LightGBM's settings were chosen on the full validation period, so this split is tilted
  slightly toward the baseline: an LLM win there is conservative.

## Known limitations
- Spring/summer only (Mar-Aug 2026); no winter peaks in test.
- Evening peak only (HE16-20); the sunset net-load ramp (HE19-21) and winter mornings (HE7-9) are natural extensions.
- Settled actual load is used as the label (ERCOT's real-time postings differ slightly).
- Claude runs go through the Claude Code CLI: temperature is not settable, the CLI adds a short preamble to the
  system prompt, and thinking is either off or adaptive (effort level recorded per run).
- Frontier training data could in principle overlap Mar-Aug 2026; dates are stripped and the target
  (hourly zonal forecast error) is not something a model would memorize.

## Commands
    uv run python -m ercot_uncertainty.eval run --backend claude-cli --model claude-opus-5-5 --split test
    uv run python -m ercot_uncertainty.eval_report results/test__*.jsonl --split test \
        --reference results/test__baseline__lightgbm-v2.jsonl --dollars
