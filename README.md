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

36 templates in 7 families (`src/ercot_bench/tasks/templates/`), difficulty 1-3:
basic aggregates, time conventions (HE vs start, UTC vs local, DST repeated/missing hours, 15-min vs hourly,
cents/kWh), spreads (DA-RT, hub/LZ basis, zone ranking), conditional joins (net load, load thresholds,
wind during negative prices, gas at the peak), events (threshold counts, longest run, peak timestamp,
day lists), forecast error, and battery arithmetic (fixed DA/RT schedules, best single cycle, monthly revenue).

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
cd environments/ercot_sql && uv sync
OPENAI_API_KEY=x uv run eval ercot-sql --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.split test_in_template --env.taskset.subset-size 8 -n 8 -r 1 \
  --model mlx-community/Qwen3-4B-4bit --client.base-url http://localhost:8089/v1 --client.api-key-var OPENAI_API_KEY \
  --sampling.max-tokens 2048 --no-rich
uv run python ../../scripts/compare_verifiers_rewards.py outputs/   # verifiers reward == our harness reward
```

Env args: `--env.taskset.split` (split name or JSONL path), `--env.taskset.subset-size`,
`--env.taskset.db-path`, `--env.taskset.task.db-path`, `--env.taskset.task.query-timeout-s`.

## SFT (Phase 5) and RL (Phase 6) with prime-rl

prime-rl has a native `uv run sft` entrypoint (a HF/local dataset with a `messages` column), so SFT and RL share
one stack. Configs in `configs/prime-rl/` validate against the current `prime-rl-configs` schema.

```bash
# 1. frontier solutions on train -> SFT data (correct only, deduped by SQL, capped per task)
uv run ercot-bench eval --backend claude-cli --model sonnet --split train --k 4
uv run ercot-bench sft-build results/train__claude-cli__sonnet.jsonl        # -> data/sft/train.jsonl

# 2. on a GPU box (CUDA required): prime-rl checkout next to this repo
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
cd prime-rl && uv pip install -e ../ercot-bench -e ../ercot-bench/environments/ercot_sql
uv run sft @ ../ercot-bench/configs/prime-rl/sft.toml --run.name ercot-sft

# 3. evaluate the SFT checkpoint with our harness (serve with prime-rl's vLLM wrapper)
uv run inference --vllm.model outputs/ercot-sft/<hf-export> --server.port 8000
uv run ercot-bench eval --backend openai-compat --model <name> --base-url http://<gpu-host>:8000/v1 --split train --k 8 --use-n
uv run ercot-bench rl-subset results/train__openai-compat__<name>.jsonl     # -> data/tasks/rl_train.jsonl

# 4. RL: smoke test first, then the real run
uv run rl @ ../ercot-bench/configs/prime-rl/rl_smoke.toml --run.name ercot-smoke   # expect non-constant reward
uv run rl @ ../ercot-bench/configs/prime-rl/rl.toml --model.name <sft checkpoint> --run.name ercot-rl
```

## Layout

```
src/ercot_bench/  ingest/ db.py schema_doc.py quality.py tasks/ env/ models/ eval/ sft/ cli.py
environments/ercot_sql/   verifiers v1 taskset package
configs/                  ercot_bench.toml, prime-rl/{sft,rl,rl_smoke}.toml
scripts/                  gridstatus probes, verifiers-vs-harness reward comparison
tests/                    pytest (time/DST, execution safety, scoring, generator determinism, report)
data/, results/           gitignored
```
