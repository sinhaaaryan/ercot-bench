# Project brief (condensed from the team's original)

Track 1 (Open Grid Data), Base Power hackathon. Judged on a 5-min demo video + codebase.

## Thesis
ERCOT publishes point load forecasts per model (+ which model is in use) but no day-specific
uncertainty. Predict HOW WRONG ERCOT's day-ahead load forecast will be (a distribution of
actual - forecast) per weather zone for peak hours, from numeric signals + NWS forecaster text.
ERCOT's model spread only reflects load-model disagreement given the same weather input; NWS
Area Forecast Discussions talk about the weather forecast itself being uncertain.
Base cares because batteries monetize uncertainty: calm day -> commit more day-ahead, uncertain
day -> hold capacity for real-time.

## Target
- actual - forecast, per weather zone (Coast, East, FarWest, North, NorthCentral, SouthCentral,
  Southern, West), peak hours. Stretch: net load.
- Forecast vintage = latest published before cutoff (10:00 Central, day before operating day).

## Inputs (strictly before cutoff)
ERCOT 7-day load forecast by model & weather zone (model spread, chosen model, revisions);
actual load; wind/solar forecasts+actuals; NWS AFDs (FWD, HGX, EWX, MAF, ...), only the latest
discussion per office covering the zone, trimmed to forecast reasoning; day-ahead weather;
context (recent errors, extreme temps, 4CP days, holidays, DST). Prompt <= ~4k tokens.

## Output
JSON only: {"hours":[{"he":17,"p10":-850,"p50":120,"p90":1400}, ...]}.
Variant: numeric baseline gives percentiles, LLM outputs widen/tighten multiplier.

## Reward
Pinball loss on p10/p50/p90, normalized per example vs numeric baseline:
reward = clip(1 - model_loss/baseline_loss, -1, 1). Invalid JSON -> fixed penalty.
Out-of-order percentiles -> sort or penalize. Unit tests: perfect prediction best; over-wide
intervals penalized on calm days; parsing edge cases.

## Baselines / eval (rolling time splits)
1 historical (zone/hour/month) 2 ERCOT model spread calibrated 3 LLM text-only RL
4 LLM spread+text RL (headline: beats #2?) 5 frontier zero-shot (Claude via `claude -p`).
Metrics: pinball, 80% interval coverage. Diagnostic: gains on days where models agreed but missed.
Frontier fairness: same prompts, fixed settings, log parse failures, cost/latency, evaluate after
training cutoffs.

## Dollar translation
Commit less DA when uncertainty high. revenue = sum DA_award*DA_price + sum_15min
(actual - DA_award)*RT_price - degradation. Base unit ~39.2 kWh / 20 kW, ~85% RTE, price taker.
Separate pre/post 2025-12-05 (RTC+B).

## Training
prime-rl + verifiers (fallback TRL GRPO LoRA). Model Qwen3-4B-Instruct-2507 (smoke with 1.7B).
LoRA r16-32, lr ~1e-5, 8 gens/prompt, max completion 256, KL ~0, temp 1.0. Optional SFT warm
start on baseline percentiles.
Log: reward mean/std, zero-variance group fraction, entropy, length, clip frac, logprob gap,
held-out pinball + calibration.

## Must-have tests
Leakage (every input ts < cutoff), reward units, settlement sanity (do-nothing=$0; deliver DA award
exactly = DA revenue), 10-step training smoke incl. checkpoint save/reload.

## Demo (5 min)
problem -> architecture + tests -> one real day (text flagged risk, models agreed, forecast missed)
-> leaderboard + calibration + $ -> how Base uses it tomorrow.
