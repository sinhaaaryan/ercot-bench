# Results

One JSONL file per (eval split, model run); one line per zone-day with the model's raw answer and its scores
(`pinball_mw`, `covered`, `width_mw`, `parse_ok`, latency, cost, served model). Report cards are regenerated from these
files with `python -m ercot_uncertainty.eval_report`.

File names: `<split>[-<prompt variant>]__<backend>__<model>.jsonl`
- splits: `test` (1,593 zone-days, Mar–Aug 2026), `test_sample400` (fixed 400 subset), `val_winter` (Dec 2025–Feb 2026)
- prompt variants: none = numbers + text; `-numeric` = numbers only; `-text` = anchor + text; `-textswap` /
  `-textswap_text` = same prompts with another day's forecaster discussion (causal control)
- `baseline__lightgbm-v1` / `-v2`: the numeric models (v2 = audited, the reference)
- `openai___hackathon-new_outputs_ercot-unc-sft2b-4b_export_merged`: Qwen3-4B + SFT (= RL's selected checkpoint, step 0)
- `openai___hackathon-new_outputs_ercot-unc-rl-4b_export_step150_merged`: RL final policy (step 150, not selected)
- `openai__Qwen_Qwen3-4B-Instruct-2507`: untrained Qwen3-4B
- `claude-cli__claude-opus-5-5`: Opus 5.5 zero-shot (thinking off)

| Report | Contents |
|---|---|
| `report_test.md` | Headline: full test, all local models + LightGBM v1/v2, text-swap controls |
| `report_test_sample400.md` | Adds Opus 5.5 (all prompt variants + text swap) |
| `report_val_winter.md` | Fall/winter holdout |
| `audit/` | LightGBM audit rounds 2–3 (validation) |
| `lgbm_v1_anchor/` | Round 1 (prompts anchored on LightGBM v1): Haiku 4.5, Opus 4.5, thinking/max-effort runs, SFT v1 |
| `logs/` | Frontier run logs |
