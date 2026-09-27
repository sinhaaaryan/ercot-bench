# ERCOT-Bench

A verifiable environment for evaluating and post-training LLMs on ERCOT grid-data questions.
The model gets a natural-language question plus the database schema, writes **one DuckDB SQL query**,
we execute it read-only, and score the result against ground truth computed by our own reference code.
One reward function (`ercot_bench.env.score.score`) is used everywhere: our eval harness, SFT filtering,
and the `verifiers` taskset used by `prime-rl`.

See `SPEC.md` for the full plan. This README covers how to run everything.

## Quickstart

```bash
uv sync                                   # Python 3.11+ (3.12 used here)
uv run ercot-bench ingest -v              # ~3 min first run; cached + resumable afterwards
uv run ercot-bench quality                # data-quality report
uv run ercot-bench schema                 # the schema text shown to models
uv run ercot-bench generate --n-per-template 30
uv run pytest -q

uv run ercot-bench check-claude-cli --model sonnet      # MUST pass before any CLI-backed eval
uv run ercot-bench eval --backend claude-cli --model sonnet --split test_in_template --limit 50 --k 3
uv run ercot-bench report                               # -> results/report/report.md + report.csv
```

## Data (Phase 1)

`ercot-bench ingest` pulls from public ERCOT sources (no API key), caches raw files under `data/raw/`,
writes monthly parquet under `data/parquet/<table>/YYYY-MM.parquet`, then builds `data/ercot.duckdb`.

| table | grain | source | coverage (as built) |
|---|---|---|---|
| `rt_spp_15min` | 15-min, 15 hubs/load zones | ERCOT annual RTM SPP archive (NP6-785-ER, MIS 13061) | 2023-01-01 .. ~last week |
| `da_spp_hourly` | hourly, 15 points | ERCOT annual DAM SPP archive (NP4-180-ER, MIS 13060) | 2023-01-01 .. ~last week |
| `load_hourly` | hourly, 8 weather zones + total | ERCOT Native Load annual workbook (via `gridstatus.get_hourly_load_post_settlements`) | through last full month |
| `fuel_mix_15min` | 15-min, 10 fuels (incl. wind, solar) | ERCOT `IntGenbyFuel{year}.xlsx` (URLs in `configs/ercot_bench.toml`) | through last full month |
| `as_prices_dam_hourly` | hourly, 5 services | ERCOT annual DAM AS MCPC archive (MIS 13091) | 2023-01-01 .. ~last week |
| `wind_forecast_hourly`, `solar_forecast_hourly`, `load_forecast_hourly` | hourly | ERCOT MIS reports 13028 / 13483 / 12311 | **only ~last 7 days** without API keys |

Conventions (also in the model-facing schema doc): every table has `interval_start_utc` / `interval_end_utc`
(naive UTC), `interval_start_local` / `interval_end_local` (naive Central Prevailing Time), `delivery_date`
(operating day), ERCOT `hour_ending` (1..24), `dst_repeated_hour` (ERCOT DSTFlag), and `interval_in_hour`
for 15-min tables. Times are built from ERCOT's native hour-ending + repeated-hour fields, so DST days have
23 / 25 hours. Prices $/MWh (AS $/MW), power MW, energy MWh.

Known data notes:
- **Forecast history needs ERCOT API keys.** Public MIS keeps only ~8 days of forecast reports. Register
  (free) at apiexplorer.ercot.com, set `ERCOT_API_USERNAME`, `ERCOT_API_PASSWORD`,
  `ERCOT_PUBLIC_API_SUBSCRIPTION_KEY` in `.env`, and re-run `ercot-bench ingest --only forecasts` to pull
  the full history through `gridstatus.ErcotAPI`. Until then the forecast-error templates only produce a
  handful of recent `test_in_template` tasks.
- We parse the annual SPP archives ourselves: `gridstatus 0.36.0`'s `get_rtm_spp` crashes on the
  current-year (partial) file. For load zones we keep ERCOT's `LZ` price type and drop the `LZEW`
  (energy-weighted) duplicates the RTM archive also contains.
- ECRS AS prices are NULL before ECRS launched (June 2023): 2.4% nulls in `as_prices_dam_hourly`.
- Fuel-mix workbook URLs are not predictable; add a new year's URL to `configs/ercot_bench.toml`.
- Settlement point list, date range, and all paths are configurable in `configs/ercot_bench.toml`.

## Tasks (Phase 2)

44 templates in 8 families (`src/ercot_bench/tasks/templates/`), difficulty 1-3:
basic aggregates, time conventions (HE vs start, UTC vs local, DST repeated/missing hours, 15-min vs hourly,
cents/kWh), spreads (DA-RT, hub/LZ basis, zone ranking), conditional joins (net load, load thresholds,
wind during negative prices, gas at the peak), events (threshold counts, longest run, peak timestamp,
day lists), forecast error, battery arithmetic (fixed DA/RT schedules, best single cycle, monthly revenue), and
`advanced` (added after Sonnet saturated the first 36): 5x16/2x16/7x8 blocks with NERC holidays, an
SOC-constrained 1 MW / 2 MWh battery optimum (DP reference), load-weighted prices, spike-event counting,
peak-net-load-hour prices, RT>DA hour share, year-over-year change, and the fall-back day's off-peak hours.

Each template samples only values that exist in the data, states the answer format explicitly, and computes
ground truth with its own SQL. Degenerate samples (ties, empty sets, zero counts) are rejected.

Splits (`src/ercot_bench/tasks/splits.py`, deterministic from `seed`): ~20% of templates held out entirely
(`test_heldout_templates`); the rest split by date (2023-2024 -> `train`, 2025+ -> `test_in_template`).
Output: `data/tasks/{train,test_in_template,test_heldout_templates}.jsonl`.

Optional: `ercot-bench generate --paraphrase-model sonnet` rewords only the descriptive part of each
question via the Claude CLI; the answer-format tail is kept verbatim and a rewrite is rejected unless all
numbers, dates and settlement point names survive.

## Environment + backends + eval (Phase 3)

- Prompt: `ercot_bench/env/prompt.py` -> `(system, user)`; user = schema doc + question. Identical for all models.
- Execution: `ercot_bench/env/execute.py` -> single SELECT/WITH only (sqlglot check), read-only DuckDB with
  `enable_external_access=false`, timeout via interrupt, row limit, structured errors, thread-safe cursors.
- Scoring: `ercot_bench/env/score.py` -> `correct` 1.0 / `wrong_answer` 0 / `sql_error` 0 / `format_error` -0.2.
  Takes the last ```sql block (ignoring `<think>...</think>`). Numeric tolerance per task; timestamps
  normalized to the task's timezone; lists compared as sets.

### Claude Code CLI backend (no API key)

`--backend claude-cli --model sonnet|opus|haiku|<id>` shells out to `claude -p` once per sample with:
`--system-prompt <ours> --tools "" --max-turns 1 --strict-mcp-config --setting-sources "" --disable-slash-commands
--no-session-persistence --output-format json`, from an empty temp cwd. Verified against Claude Code 2.1.281
(`--max-turns` is accepted though not listed in `--help`). `ercot-bench check-claude-cli` runs one task with
`stream-json --verbose` and asserts: no tools available, no MCP servers, no tool calls, exactly one assistant
turn, temp cwd.

Caveat: the CLI still prepends a short fixed preamble to our system prompt (an Agent-SDK identity line, an
environment block with cwd/platform/date, and account context). `--bare` removes it but requires an API key.
It contains no tools, repo, or data, so it doesn't make Claude agentic, but the prompt is not byte-identical
to what open models see.

Temperature can't be set via the CLI (`temperature: "cli-default"` is recorded). Usage/rate-limit errors
back off exponentially; persistent limits stop the run cleanly (exit code 2). Rerun the same command to resume.

### Open models (OpenAI-compatible)

```bash
# Ollama
ollama serve & ollama pull qwen3:4b
uv run ercot-bench eval --backend openai-compat --model qwen3:4b --base-url http://localhost:11434/v1 \
  --split test_in_template --limit 50 --k 5 --no-thinking

# vLLM (local or rented GPU)
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 16384
uv run ercot-bench eval --backend openai-compat --model Qwen/Qwen3-8B --base-url http://<host>:8000/v1 \
  --split test_in_template --k 10 --use-n --concurrency 32 --no-thinking

# Apple Silicon without Ollama (what this prototype was tested with)
uvx --from mlx-lm mlx_lm.server --model mlx-community/Qwen3-4B-4bit --port 8089
uv run ercot-bench eval --backend openai-compat --model mlx-community/Qwen3-4B-4bit \
  --base-url http://localhost:8089/v1 --split test_in_template --limit 29 --k 2 --no-thinking --concurrency 2
```

`--no-thinking` sends `chat_template_kwargs.enable_thinking=false` (Qwen3). Optional API-key backend:
`--backend anthropic-api` (`uv sync --extra api`, `ANTHROPIC_API_KEY`).

### Report

`ercot-bench report [files...]` -> per model x split: pass@1, unbiased pass@5/pass@10 (when k allows),
consistency (all k correct), never-solved count, outcome breakdown, latency/tokens/cost, and breakdowns by
family, difficulty and template. Markdown + CSV in `results/report/`.

## verifiers taskset (Phase 4)

`environments/ercot_sql/` is a **verifiers v1** taskset (`ErcotSqlTaskset`). The spec assumed the legacy
`load_environment()` / `SingleTurnEnv` / rubric API; current verifiers (0.3.1) deprecates that in favor of
tasksets + harnesses, and current prime-rl consumes v1 tasksets, so we use v1. Single-turn = the tool-less
`null` harness. The reward calls `ercot_bench.env.score.score` directly (no reimplementation).

```bash
# inside the prime-rl venv (its bundled verifiers is newer than PyPI 0.3.1; flag names below match it)
cd prime-rl && uv pip install --no-deps -e ../ercot-bench -e ../ercot-bench/environments/ercot_sql
uv pip install duckdb sqlglot python-dotenv typer tzdata
VLLM_KEY=x uv run eval ercot-sql --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.split test_in_template --env.taskset.subset-size 32 -n 32 -r 2 \
  --model Qwen/Qwen3-8B --client.base-url http://localhost:8000/v1 --client.api-key-var VLLM_KEY \
  --sampling.max-completion-tokens 2048 --sampling.extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --output-dir ../ercot-bench/outputs
uv run python ../ercot-bench/scripts/compare_verifiers_rewards.py ../ercot-bench/outputs/   # verifiers == harness
```

Verified on an RTX 5090: 64 single-turn rollouts, 0 errors, and all 64 traced rewards equal our harness's
`score()` on the same replies. `environments/ercot_sql/tests/test_rewards_match.py` replays every recorded
harness completion through the taskset's `@vf.reward` as well.

Env args: `--env.taskset.split` (split name or JSONL path), `--env.taskset.subset-size`,
`--env.taskset.db-path`, `--env.taskset.task.db-path`, `--env.taskset.task.query-timeout-s`.

## SFT (Phase 5) and RL (Phase 6) with prime-rl

prime-rl has a native `uv run sft` entrypoint (a HF/local dataset with a `messages` column), so SFT and RL share
one stack. All configs in `configs/prime-rl/` validate against the installed prime-rl config schema.

| config | purpose |
|---|---|
| `sft_1gpu_qwen3_1p7b.toml` | **tested**: full fine-tune of Qwen3-1.7B on one 32 GB GPU (exportable to HF) |
| `rl_lora_1gpu.toml` | **tested**: memory-safe single-GPU LoRA RL from the SFT checkpoint, external vLLM on the same GPU |
| `rl_smoke_1gpu.toml` | single-GPU full-FT smoke config (needs more host RAM than 40 GB for 1.7B; see safeguards) |
| `sft.toml`, `rl.toml`, `rl_smoke.toml` | multi-GPU defaults (Qwen3-8B + LoRA; 1 inference + 1+ trainer GPU) |

### Tested single-GPU workflow (RTX 5090 32 GB, 40 GB RAM, WSL2)

```bash
# prime-rl checkout next to this repo (submodules over HTTPS if you have no GitHub SSH key)
git clone https://github.com/PrimeIntellect-ai/prime-rl.git && cd prime-rl
git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive
uv sync --all-extras
uv pip install --no-deps -e ../ercot-bench -e ../ercot-bench/environments/ercot_sql
uv pip install duckdb sqlglot python-dotenv typer tzdata
export VLLM_USE_V2_MODEL_RUNNER=0     # WSL2: vLLM's V2 model runner needs UVA/pinned memory, unavailable there
../ercot-bench/scripts/mem_guard.sh & # watchdog: pauses ercot-* jobs before RAM/VRAM run out (see below)

# 1. frontier solutions on train -> SFT data (correct only, deduped by SQL, capped per task)
cd ../ercot-bench
uv run ercot-bench eval --backend claude-cli --model sonnet --split train --limit 450 --k 2
uv run ercot-bench sft-build results/train__claude-cli__sonnet.jsonl        # -> data/sft/train.jsonl

# 2. SFT (~18 min on a 5090), then export DCP -> HF safetensors
cd ../prime-rl
../ercot-bench/scripts/guarded_run.sh sft 34G uv run sft @ ../ercot-bench/configs/prime-rl/sft_1gpu_qwen3_1p7b.toml \
  --data.name ../ercot-bench/data/sft --run.name ercot-sft-1p7b
uv run python tools/convert_dcp_to_bf16.py outputs/ercot-sft-1p7b/checkpoints/step_80   # -> .../weights
W=outputs/ercot-sft-1p7b/checkpoints/step_80/weights

# 3. evaluate the SFT model with our harness, build the RL prompt set
../ercot-bench/scripts/guarded_run.sh infer 10G uv run inference --vllm.model $W --vllm.gpu-memory-utilization 0.85
cd ../ercot-bench
uv run ercot-bench eval --backend openai-compat --model $W --base-url http://localhost:8000/v1 --split test_in_template \
  --limit 70 --k 4 --use-n --no-thinking --out results/test_in_template__openai-compat__qwen3-1.7b-sft.jsonl
uv run ercot-bench eval ... --split train --limit 450 --out results/train__openai-compat__qwen3-1.7b-sft.jsonl
uv run ercot-bench rl-subset results/train__openai-compat__qwen3-1.7b-sft.jsonl     # -> data/tasks/rl_train.jsonl
systemctl --user stop ercot-infer.scope

# 4. RL (LoRA) with vLLM co-located on the same GPU at 35% memory
cd ../prime-rl
../ercot-bench/scripts/guarded_run.sh infer 10G uv run inference --vllm.model $W --vllm.gpu-memory-utilization 0.35 \
  --vllm.max-model-len 8192 --vllm.enable-lora --vllm.max-lora-rank 16 &
../ercot-bench/scripts/guarded_run.sh rl 20G uv run rl @ ../ercot-bench/configs/prime-rl/rl_lora_1gpu.toml \
  --model.name $W --run.name ercot-rl-lora
```

### Memory safeguards (why they exist)

A first single-GPU RL attempt (full fine-tune of the 1.7B SFT model) exhausted the 40 GB of host RAM during the
first trainer step and took down the whole WSL VM. Cause: prime-rl keeps the optimizer state in host RAM by default
(`trainer.model.optim_cpu_offload = true`), ~20-27 GB of Adam state for a 1.7B full fine-tune, on top of vLLM, the
orchestrator, env workers, and DuckDB (whose default memory limit is 80% of RAM per process), with no swap.
Now:

- `scripts/guarded_run.sh <name> <cap> <cmd>` runs each job in its own `systemd --user` scope (`ercot-<name>.scope`)
  with `MemoryMax=<cap>`, `MemoryHigh=90%`, `MemorySwapMax=0`: an OOM kills only that job, never the VM.
- `scripts/mem_guard.sh` (watchdog, log `/hackathon/mem_guard.log`): **pauses** (cgroup freeze) all `ercot-*` scopes
  when host MemAvailable < 6 GB or GPU memory > 31.8 GB, resumes them above 9 GB, and stops the largest scope if
  MemAvailable < 2.5 GB. Thresholds are env vars. Pause/resume manually with `systemctl --user freeze|thaw`.
- DuckDB is capped per process (`ERCOT_DUCKDB_MEMORY_LIMIT`, default 1 GB; `ERCOT_DUCKDB_THREADS`, default 2).
- RL uses LoRA with the optimizer on the GPU (`rl_lora_1gpu.toml`). In the verified run host RAM never dropped below
  27 GB available and VRAM peaked at 21.8 / 32 GB (vLLM 35% + trainer).

## Results (RTX 5090 + Claude Code CLI)

Same stratified test subsets for every model (test_in_template: 70 tasks, 2/template; held-out templates: 27 tasks,
never seen in training). pass@1 = mean accuracy over k samples (k=3 Claude/Qwen3-8B base, k=4 others);
"consistent" = correct on all k samples. Open models: Qwen3 with thinking off, K2 with `reasoning_effort=low`.
All fine-tunes use the same 802 correct Sonnet solutions (SFT only, no RL yet).

| model | test_in_template | held-out templates | consistent (in-template) | mean output tokens |
|---|---|---|---|---|
| Claude Sonnet (claude-cli) | 98.1% | 87.7% | 97.1% | 218 |
| Claude Haiku 4.5 (claude-cli) | 97.6% | 85.2% | 92.9% | 2,949 (reasoning) |
| **Qwen3-8B + SFT (LoRA)** | **95.0%** | 58.3% | 88.6% | 109 |
| **K2-Horizon-7B + SFT (LoRA)** | 91.8% | **63.0%** | 82.9% | 101 |
| Qwen3-1.7B + SFT (full FT) | 83.2% | 53.7% | 74.3% | 111 |
| K2-Horizon-7B base | 66.8% | 45.4% | 31.4% | 407 (reasoning) |
| Qwen3-8B base | 52.9% | 37.0% | 40.0% | 245 |
| Qwen3-1.7B base | 16.8% | 13.9% | 10.0% | 282 |

Training on one RTX 5090 (all under the memory safeguards):

| run | recipe | time | peak VRAM |
|---|---|---|---|
| Qwen3-1.7B SFT | prime-rl, full FT, 80 steps | 18 min | 29.2 GB |
| Qwen3-8B SFT | prime-rl, LoRA r32, bf16 params, 40 steps (~3.2 epochs) | 30 min | 20.0 GB |
| K2-Horizon-7B SFT | `scripts/sft_lora_hf.py`, LoRA r32, 2 epochs | 14 min | 20.1 GB |
| Qwen3-1.7B RL smoke | prime-rl GRPO, LoRA, 15 steps | 2.5 min | 21.8 GB (incl. vLLM) |

- Fine-tuned 7-8B models get within ~3-6 points of Claude on in-template questions while answering in ~100 tokens
  (single-request latency ~1-2 s on the 5090 vs ~7-25 s for Haiku, which reasons for ~3k tokens), self-hosted.
- Generalization to unseen question types is the open gap (58-63% vs Claude's 85-88%): the next lever is more
  template diversity and RL, not more epochs.
- The stronger base (K2) generalizes better after SFT (63.0% held-out) even though Qwen3-8B wins in-template.
- RL smoke (LoRA GRPO from the 1.7B SFT checkpoint): per-step reward 0.38-0.88, non-constant, 0 errors.
- verifiers `eval` on `ercot-sql` vs our harness: 64/64 rewards identical.
- **Flag (per spec):** Claude is above the ~85% threshold on both splits; templates need to get harder to track
  frontier progress.

K2-Horizon-7B notes: served by vLLM's Transformers backend (`--vllm.trust-remote-code True`); prime-rl SFT needs a
typed renderer K2 doesn't have, so it's trained with `scripts/sft_lora_hf.py` (LoRA without peft; loss only on
assistant tokens; LM head applied only to answer positions, which cut peak VRAM from >32 GB to 20 GB given K2's
250k vocab). Its chat template requires a thinking field, so `data/sft_k2` sets `think_faster=""` and the tuned
model learns to answer immediately under `reasoning_effort=low`.

## Layout

```
src/ercot_bench/  ingest/ db.py schema_doc.py quality.py tasks/ env/ models/ eval/ sft/ cli.py
environments/ercot_sql/   verifiers v1 taskset package
configs/                  ercot_bench.toml, prime-rl/{sft,rl,rl_smoke}.toml
scripts/                  gridstatus probes, reward comparison, guarded_run.sh + mem_guard.sh safeguards
tests/                    pytest (time/DST, execution safety, scoring, generator determinism, report)
data/, results/           gitignored
```
