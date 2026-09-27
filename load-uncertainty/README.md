# ERCOT load-forecast uncertainty: can forecaster text beat a strong numeric model?

**Short answer: the text carries real, day-specific signal, but a tuned numeric model already captures what's usable.**
We built the eval, the numeric model (which is the product Base could use tomorrow), and a rigorous test of whether
LLMs reading NWS forecaster discussions add value on top of it: frontier models zero-shot, and a 4B model trained
with SFT + RL.

**Track 1 (Open Grid Data).** ERCOT publishes a *point* day-ahead load forecast, one number per zone and hour from
whichever of its 8 models is "in use", and nothing about how uncertain a given day is. We built:

1. **An uncertainty eval.** Predict the p10/p50/p90 of ERCOT's day-ahead load-forecast error for each weather zone
   and evening-peak hour, using only information public before the 10:00 CT day-ahead deadline.
2. **A non-LLM uncertainty model** (LightGBM quantile regression), audited in three rounds. It is calibrated, its median
   makes ERCOT's official forecast miss about 19% smaller (full test), and Base could run it tomorrow.
3. **A frontier benchmark.** Claude models, zero-shot, on the same eval, including a causal *text-swap* test of
   whether NWS forecaster discussions carry day-specific signal.
4. **SFT + RL of a small open model** (Qwen3-4B-Instruct-2507, prime-rl + Verifiers), trained against the eval's own
   scoring rule to beat the non-LLM model by reading the forecaster text.

> **Why Base should care:** uncertainty is what a battery fleet and a retail book monetize or hedge. On a calm day you
> can commit more day-ahead; on an uncertain one you hold capacity or hedge volume. ERCOT doesn't tell you which day
> is which. This repo does.

## Headline results (test set: 1,593 zone-days, Mar–Aug 2026, strictly after all training data)

Pinball loss (lower is better) vs the LightGBM v2 reference; 95% CIs are day-clustered bootstraps.

| Model | Pinball (MW) | vs LightGBM v2 | Notes |
|---|---|---|---|
| **LightGBM v2** (our non-LLM model) | **127.3** | reference | 82% coverage of the 80% interval |
| LightGBM v1 (before audit) | 133.5 | +4.9% [+2.8, +7.1] | |
| Qwen3-4B, untrained | 170.6 | +34% [+29, +39] | intervals 55% too wide |
| Qwen3-4B + SFT | 127.6 | +0.2% [−0.1, +0.5] | ties the anchor, as designed |
| Qwen3-4B + SFT + RL (selected: best validation checkpoint = step 0) | 127.6 | +0.2% [−0.1, +0.5] | RL never beat the SFT start on validation |
| Qwen3-4B + SFT + RL, final policy (step 150, not selected) | 127.3 | 0.000% [−0.0, +0.0] | converged to reproducing LightGBM v2 |

Fall/winter holdout (`val_winter`, 810 zone-days, Dec 2025 – Feb 2026): LightGBM v2 beats v1 by **13.1%** [8.5, 18.2];
SFT −0.2% and RL (step 150) −0.004% vs LightGBM v2 (ties).

On the 400-example sample (162 days), Opus 5.5 zero-shot:

| Opus 5.5 given LightGBM v2's percentiles + … | Pinball (MW) | vs LightGBM v2 |
|---|---|---|
| the real day's forecaster discussion (text only) | 113.7 | −0.8% [−2.2, +0.5] |
| a **different** day's discussion (text swap) | 117.1 | +2.2% [+1.0, +3.3] |
| ERCOT numbers + weather only | 116.6 | +1.7% [−0.1, +3.4] |
| numbers + text | 116.7 | +1.8% [0.0, +3.7] |
| LightGBM v2 alone | 114.6 | reference |

**The real day's text beats a swapped day's text by 2.9% [−4.9, −1.2]** (paired, significant). The discussions
carry day-specific signal and the model uses it. But even the strongest frontier model, zero-shot, only ties the
tuned tree model. It helps on the days the thesis targets (ERCOT models agreed: −1.6%; heavy rain: −1.6%; heat: −1.3%)
and hurts when it second-guesses the numbers.

## What RL learned (and why that's a result, not a failure)

RL (GRPO, reward = MW of pinball loss saved vs LightGBM v2) ran 155 steps before an approved early-stop rule fired
(entropy dipped below 0.10; validation reward had been flat at 0.0000 for three evals). The policy converged to
**reproducing LightGBM v2's percentiles**: its small SFT-induced deviations were net-negative, so RL removed them.
The **text-swap test** confirms it: swapping in another day's forecaster discussion changes the RL policy's loss by
0.00% (SFT: +0.04%). By contrast, Opus 5.5 zero-shot is significantly hurt by the wrong day's text (−2.9% real vs
swapped), so the discussions *do* carry day-specific signal. But against a tuned numeric model that already prices in
ERCOT's model disagreement and the weather, that signal was too weak or too inconsistent for a 4B model to exploit
reliably in 155 RL steps. Deferring to the strong baseline is the reward-optimal behaviour.

**Versus ERCOT's own forecast:** ERCOT publishes no uncertainty at all. On the full test set, adding LightGBM v2's
predicted median error to ERCOT's official in-use forecast makes the average evening-peak miss **18.8% smaller**
(393 vs 484 MW), and its p10/p50/p90 is **17% better** (pinball) than a same-range-every-day uncertainty.

Where that comes from (not weather: weather alone adds only 1.3%). ERCOT publishes 8 model forecasts but uses one:

| Correction to ERCOT's 09:30 forecast (full test, evening peak) | Mean miss | vs ERCOT |
|---|---|---|
| ERCOT's official in-use forecast | 483.7 MW | — |
| always use ERCOT's best single model on train | 473.4 MW | −2.1% |
| subtract the last 90 days' median miss (persistent zonal bias) | 444.5 MW | −8.1% |
| **weighted average of ERCOT's own 8 models** (linear, fit on train) | 423.8 MW | **−12.4%** |
| LightGBM v2 median (both, plus context, learned jointly) | 392.7 MW | −18.8% |

Caveat: this beats the forecast available at the 10:00 day-ahead deadline; ERCOT's later intraday updates are better.

## What we see that others miss

- **ERCOT's own model choice is the biggest signal.** How far each of ERCOT's 8 models sits from the in-use forecast,
  and which model is in use, are LightGBM's strongest inputs (24% of split gain for in-use-vs-median alone). They grew
  more valuable as ERCOT's newer model **X** (since Aug 2024) accumulated history: −10% loss on validation.
- **A third of big misses happen when ERCOT's models agree** (33% of top-quintile errors), exactly where model
  spread says "calm". That's the gap forecaster text is meant to fill.
- **4CP curtailment:** on summer weekdays when ERCOT's forecast peak tops the month's highest actual so far, load
  comes in **1,543 MW below forecast** on average (655 days). Large loads curtail to dodge transmission charges.
- **ERCOT's errors are skewed and fat-tailed:** extreme misses happen 2.7× more often than a normal distribution
  predicts, so we forecast percentiles, not mean ± σ.
- **Load uncertainty did not predict real-time price premiums in 2026** (post RTC+B). The evening RT−DA premium
  averaged about $0 and correlated ≤ 0.05 with any predicted or even realized load error. Battery commit/hold rules
  keyed on load uncertainty lose money in this period (documented negative result).

## Dollar translation (retail load hedging)

A load-serving entity that buys day-ahead **ERCOT's forecast + our predicted median error** instead of ERCOT's
forecast alone (evening peak, full test set, per GW of load per zone):

| Median from | Energy settled at real-time prices | Exposure to RT−DA price swings |
|---|---|---|
| LightGBM v2 | **−33%** | **−36%** |
| Qwen3-4B + SFT | −33% | −36% |
| LightGBM v1 | −29% | −32% |

Expected imbalance P&L in this period is about zero (RT ≈ DA on average). The value is risk reduction on the days
real-time prices spike. See `src/ercot_uncertainty/sim/imbalance.py`.

## Quickstart

```bash
uv sync
uv run pytest -q                                     # 95+ tests: leakage, reward, settlement, env
# evaluate any OpenAI-compatible model (e.g. your own vLLM server) on the eval:
uv run python -m ercot_uncertainty.eval run --backend openai --model <model> --base-url http://localhost:8300/v1 --split test
# or Claude via Claude Code:
uv run python -m ercot_uncertainty.eval run --backend claude-cli --model claude-opus-5-5 --split test_sample400
# report card (pinball, zone-normalized skill, calibration, thesis slices, value metrics), vs the declared reference:
uv run python -m ercot_uncertainty.eval_report results/test__*.jsonl --split test --reference results/test__baseline__lightgbm-v2.jsonl
```

**Plug in your own policy:** anything that returns `{"hours":[{"he":16,"p10":…,"p50":…,"p90":…}, …]}` for a prompt can
be scored. Serve it behind an OpenAI-compatible API, or write its JSON answers into a results file and run
`eval_report`. To train against the same reward, use the Verifiers environment in `environments/ercot_uncertainty_env`
(see `docs/TRAINING.md`).

Rebuilding the data from public sources (ERCOT public API keys needed for forecast history) is covered in `DATA.md`.

Per-run results and how to read the file names: `results/README.md`.

## How it works

| Part | Where | Key design choices |
|---|---|---|
| Data | `src/ercot_uncertainty/data/` | ERCOT 7-day load forecast by model and zone (NP3-565-CD, the 09:30 post before the 10:00 cutoff); actual zonal load; 74,918 NWS Area Forecast Discussions (9 offices, IEM); Open-Meteo day-2 weather (GFS/ECMWF/ICON, 19 points) |
| Leakage guards | `features/build.py`, `tests/test_leakage.py` | Every input's publish/issue time asserted < cutoff; recent errors only through D−2 (ERCOT posts zonal actuals once a day); dates stripped from forecaster text; independent review in `docs/REVIEW.md` |
| Eval | `eval.py`, `eval_report.py`, `docs/EVAL.md` | Pinball loss (proper scoring rule); zone-normalized skill; per-percentile and per-zone calibration; day-clustered CIs; thesis slices; headline comparison declared before RL |
| Baseline | `baselines/numeric.py`, `scripts/audit_baseline*.py` | LightGBM quantile regression, 500 trees/15 leaves, per-model deviations + in-use model, per-zone out-of-fold conformal calibration. Beat XGBoost, CatBoost, quantile forests, and ensembles |
| Frontier | `eval.py --backend claude-cli` | Identical prompts; thinking off/on/max recorded; served model verified per call |
| SFT | `configs/prime-rl/sft_4b_lora_1gpu_v2.toml` | Template-exact renderer (`prime-qwen3`); jittered anchor targets so the policy keeps exploring (v1 collapsed to copying) |
| RL | `configs/prime-rl/rl_4b_lora_1gpu.toml`, `environments/` | GRPO (prime-rl), LoRA r32, reward = pinball MW saved vs LightGBM v2 in units of the zone's typical loss (aligned with the eval), lr 5e-5 with warmup and linear decay, best checkpoint by validation |

## Limitations

- Test covers spring/summer 2026 only; a fall/winter holdout (`val_winter`) is reported separately.
- Evening peak (HE16–20) only; the sunset ramp (HE19–21) and winter mornings (HE7–9) are natural extensions.
- Settled actual load is used as the label; ERCOT's real-time postings differ slightly.
- Claude runs go through the Claude Code CLI (temperature not settable; small system-prompt preamble).
- Detectable effects: about ±2% on the full test set, about ±3.4% on the 400 sample.

## Also tried (see `results/lgbm_v1_anchor/`)

Haiku 4.5 and Opus 4.5 zero-shot (4–5% worse than the anchor); thinking modes (Opus 5.5 max effort, about 20k
thinking tokens per answer, no measurable gain); SFT v1 (chat-template bug + exact-copy targets collapsed exploration);
battery commit/hold rules (lose money: see above).

## Future work

1. **One-knob output (widen/narrow).** Instead of 15 numbers (p10/p50/p90 × 5 hours), the model outputs one scale
   factor for LightGBM's interval (`{"scale": 1.3}`). A far easier, lower-noise learning problem, and the variant in the
   original design. Parsing and scaling already exist in `task.py`; estimated ~2 h for prompt/env + SFT + 100 RL steps.
2. **Text-only prompts and thesis-day training.** Opus 5.5 did best with anchor + text only, and the text helped most
   on days ERCOT's models agreed and on storm/heat/rain days: train RL there.
3. **Multi-turn tool-using environment:** the agent chooses which analog days, neighbouring offices, or model track
   records to consult before answering.
4. Net load (wind/solar), the sunset ramp (HE19–21), winter mornings, and an all-season test year as data accrues.

Note on power: the full test set detects differences of about 2%; the plausible text effect is about 0–1%, so a
convincing LLM win would also need more test days.
