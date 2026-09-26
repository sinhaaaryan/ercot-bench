# ERCOT-Bench: Build Spec (v1)

## What we're building

A verifiable environment for training and evaluating LLMs on ERCOT grid-data questions. The model is given a natural-language question plus a database schema, writes one DuckDB SQL query, we execute it, and we score the result against ground truth computed by our own reference code.

The same environment serves three purposes:
1. **Benchmark**: compare frontier models vs. open-source models on ERCOT data tasks.
2. **SFT data source**: collect correct frontier-model solutions to fine-tune an open model.
3. **RL environment**: packaged as a Prime Intellect `verifiers` environment and trained with `prime-rl`.

Context: this is for a Base Power hackathon (Track 1: Open Grid Data, Track 3: Most Commercializable). The pitch is that a small open model, post-trained on this environment, matches or beats frontier models on ERCOT-specific analysis at a fraction of the cost, and can be self-hosted.

## Guiding principles

- **Single-turn first.** Question in, one SQL query out, execute, score. No multi-turn agent loop in v1.
- **Rewards must be deterministic.** Every answer is computed by code from the data. No LLM judging.
- **One reward function, used everywhere.** Our eval harness, SFT filtering, and the `verifiers` rubric all call the same `score()` code.
- **Everything configurable, nothing hardcoded** for model names, endpoints, date ranges, and settlement points.
- **No API keys assumed.** Frontier models are accessed through the Claude Code CLI (subscription login). Open models are served locally. API-key clients are optional extras.
- **Stop and report at the end of each phase** (see Phases). Show what works, what's broken, and sample outputs before moving on.

## Tech stack

- Python 3.11+, `uv` for dependency management
- `gridstatus` for pulling ERCOT data
- DuckDB as the query engine, Parquet as storage
- `polars` or `pandas` for ingestion transforms
- `pydantic` for task/result schemas
- `typer` for the CLI
- `pytest` for tests
- Prime Intellect stack for RL: `verifiers` (environment), `prime` CLI, `prime-rl` (trainer)
- Model backends:
  - Claude Code CLI (`claude -p`) for frontier-model evals, no API key
  - Any OpenAI-compatible endpoint for open models (Ollama or vLLM, local or on a rented GPU)
  - Optional: Anthropic/OpenAI SDK clients if keys are later available (`.env`)

**The Prime Intellect stack moves quickly.** Before writing any `verifiers` or `prime-rl` code, read the current docs (https://docs.primeintellect.ai/verifiers and the prime-rl docs) and the installed library source. Use whatever environment style and config format is currently recommended rather than assuming from memory.

## Repo layout

```
ercot-bench/
  SPEC.md
  pyproject.toml
  .env.example
  data/                    # gitignored: raw/, parquet/, ercot.duckdb
  src/ercot_bench/
    ingest/                # pulling + normalizing ERCOT data
    db.py                  # DuckDB connection, schema creation, read-only access
    schema_doc.py          # generates the schema description shown to models
    tasks/
      templates/           # one module per question template family
      generator.py         # samples params, builds questions + ground truth
      splits.py
    env/
      prompt.py            # builds the model prompt
      execute.py           # safe SQL execution
      score.py             # THE reward function
    models/
      base.py              # shared client interface
      claude_cli.py        # Claude Code headless backend
      openai_compat.py     # Ollama / vLLM / any OpenAI-compatible endpoint
    eval/
      run.py               # run a model over a task set, k samples per task
      report.py            # aggregate metrics, tables
    sft/                   # build SFT dataset from frontier outputs
  environments/
    ercot_sql/             # verifiers environment module (wraps env/ + tasks/)
  configs/
    prime-rl/              # prime-rl training configs
  tests/
  results/                 # gitignored: eval outputs as JSONL
```

## Phase 1: Data layer

**Goal:** a local DuckDB database with clean, well-documented ERCOT tables.

Datasets (start with 2023-01-01 through the latest available date):
- Real-time settlement point prices (15-minute)
- Day-ahead settlement point prices (hourly)
- System load, actual and forecast (by weather zone if available)
- Wind actual and forecast
- Solar actual and forecast
- Fuel mix / generation by fuel type
- Ancillary service prices (day-ahead), if easily available

Scope the settlement points to hubs and load zones only (e.g. `HB_HOUSTON`, `HB_NORTH`, `HB_SOUTH`, `HB_WEST`, `HB_BUSAVG`, `HB_HUBAVG`, `LZ_HOUSTON`, `LZ_NORTH`, `LZ_SOUTH`, `LZ_WEST`, `LZ_AEN`, `LZ_CPS`, `LZ_LCRA`, `LZ_RAYBN`). No resource nodes in v1.

Requirements:
- **Verify `gridstatus` method names and signatures against the installed version.** Do not assume them from memory. Inspect the library and write a tiny probe script first.
- Cache raw pulls to disk so re-runs don't re-download. Ingestion must be resumable (pull by month, skip months already on disk).
- Normalize every table to include: `interval_start_utc`, `interval_end_utc`, `interval_start_local` (America/Chicago), and the native ERCOT fields needed to reason about conventions (e.g. hour-ending, DST flag) where the source provides them.
- Units: prices in $/MWh, load/generation in MW. Put units in column names or the schema doc, never ambiguous.
- Write `schema_doc.py` to generate a concise text description of every table and column (name, type, units, meaning, time convention). This text goes into model prompts, so keep it tight and accurate.
- Include a data-quality check script: row counts per month, gaps, duplicates, the DST transition hours, null rates. Print a summary.

**Done when:** `ercot-bench ingest` builds `data/ercot.duckdb`, the quality report is clean or known issues are documented, and `ercot-bench schema` prints the schema doc.

## Phase 2: Task generator

**Goal:** thousands of questions with exact, code-computed answers.

Each template is a small class/module with:
- `id`, `family`, `difficulty` (1–3)
- `sample_params(rng)`: picks settlement point, date range, thresholds, etc. Only picks values that exist in the data.
- `render_question(params) -> str`: the natural-language question
- `reference_answer(params, conn)`: computes ground truth with our own SQL or Python (not shown to the model)
- `answer_type`: one of `number`, `integer`, `timestamp`, `category`, `list`
- `tolerance`: for numeric answers (absolute and/or relative)

Question wording must state the required answer format explicitly (units, rounding, timezone, e.g. "Answer in $/MWh, rounded to 2 decimals" or "Give the interval start in Central time").

Start with ~20–30 templates across these families, spread across difficulty levels:
- **Basic aggregates**: average/max/min price at a location over a period.
- **Time conventions**: questions that depend on hour-ending vs. interval start, local vs. UTC, the DST repeated/missing hour, 15-min vs. hourly granularity.
- **Spreads and comparisons**: DA vs. RT spread, hub vs. load zone basis, which zone had the highest average price.
- **Conditional/joined**: price statistics during hours meeting a condition on another table (e.g. net load above a threshold, where net load = load − wind − solar).
- **Event finding**: count of intervals above a price threshold, the timestamp of the peak, longest consecutive run above a threshold.
- **Forecast error**: wind/solar/load forecast error statistics, and price behavior when forecast error is large.
- **Battery arithmetic**: revenue of a simple fixed charge/discharge schedule for a battery of given size and efficiency at a given location and day.

Splits (`splits.py`):
- Hold out entire templates for a `test_heldout_templates` split (~20% of templates).
- For the remaining templates, split by date: questions about 2023–2024 go to `train`, questions about 2025+ go to `test_in_template`.
- Split assignment is deterministic from a seed.

Optional: a paraphrase step that uses an LLM to reword questions for variety, **only if** the answer-format instructions are preserved verbatim. Keep it behind a flag.

Output: JSONL files per split. Each task has `task_id`, `template_id`, `difficulty`, `question`, `params`, `answer`, `answer_type`, `tolerance`.

**Done when:** `ercot-bench generate --n-per-template N` produces all splits, and a test verifies that every template's reference answer is non-null and deterministic across runs.

## Phase 3: Environment, model backends, eval harness

### Prompt, execution, scoring

**Prompt (`env/prompt.py`)**: returns a `(system_prompt, user_prompt)` pair. System: instructions. User: schema doc + question. The model must return exactly one SQL query in a ```sql fenced block. The query must return exactly one row and one column containing the answer (or one column of values for `list` answers). The prompt is identical for every model and backend.

**Execution (`env/execute.py`)**:
- Read-only DuckDB connection
- Query timeout and a row limit
- Reject anything that isn't a single `SELECT`/`WITH` statement
- Capture errors as structured results, never crash the run
- Must be safe to call concurrently (it will be called from async RL rollouts in Phase 5)

**Scoring (`env/score.py`)**: returns a result object with `reward` and `outcome`:
- `correct` → reward 1.0
- `wrong_answer` → 0.0
- `sql_error` → 0.0
- `format_error` (no parseable SQL block, wrong result shape) → −0.2
- Numeric comparison uses the task's tolerance. Timestamps compared after timezone normalization. Lists compared as sets unless order matters for the template.

### Model backends (`models/`)

One interface: `generate(system_prompt, user_prompt, n) -> list[Completion]`, where each completion records text, token counts (if available), latency, and backend metadata.

**`claude_cli.py`: frontier models via Claude Code headless mode (no API key)**

Shell out to `claude -p` once per sample, using the logged-in subscription. The key requirement: **Claude Code must behave as a plain single-turn model, not an agent.** Otherwise it could run the SQL itself and fix mistakes, which inflates its score. So:
- Replace Claude Code's default system prompt with ours (`--system-prompt` or `--system-prompt-file`)
- Disable all tools (e.g. `--tools ""` or the current equivalent)
- Limit to one turn (`--max-turns 1`)
- Run the subprocess with `cwd` set to an empty temp directory, so the repo and data are not visible
- Use `--output-format json` and read the response text from the result field
- Model selectable via `--model` (configurable, e.g. `sonnet`, `opus`)

**Verify every flag name against `claude --help` for the installed version before relying on it.** Then write a sanity check command (`ercot-bench check-claude-cli`) that runs one task with `--output-format stream-json --verbose` and asserts that no tool calls occurred and exactly one assistant turn was produced. Run this before any large eval.

Operational notes:
- Configurable concurrency, default 4. Subscription rate limits apply.
- Detect usage-limit / rate-limit errors, back off, and if limits persist, stop cleanly with a clear message. The runner is resumable, so a later rerun continues where it stopped.
- Temperature can't be set through the CLI. Record `temperature: "cli-default"` in results.
- Token counts and cost may be partially available from the JSON output; record what's there, leave the rest null.

**`openai_compat.py`: open models**

Standard OpenAI-compatible chat client with configurable `base_url`, model name, temperature, and `n`. Must work with:
- Ollama locally (`http://localhost:11434/v1`), for quick tests with small Qwen models
- vLLM (local or rented GPU), for serious evals and for evaluating trained checkpoints

Document the exact commands to serve a model with each in the README.

### Eval runner (`eval/run.py`)

- Runs a model over a split with `k` samples per task (default k=10)
- Supports `--limit N` to run on a subset (start with ~50 tasks for CLI-backed runs)
- Concurrency with retries
- Writes one JSONL line per sample to `results/` (task, model, backend, completion, SQL, result, outcome, reward, tokens, latency)
- Resumable: skips (task, sample index) pairs already present in the output file

### Report (`eval/report.py`)

Per model and split:
- **pass@1**: mean accuracy across all k samples
- **pass@k**: fraction of tasks solved by at least one of k samples (use the standard unbiased estimator; report pass@1, pass@5, pass@10 where k allows)
- **Consistency**: fraction of tasks correct in all k samples
- **Never solved**: count of tasks with 0/k correct. These give no RL signal and indicate templates that may be too hard or broken.
- Breakdown by template family and difficulty
- Outcome breakdown (wrong answer vs. SQL error vs. format error)
- Mean latency, and tokens/cost where available
- Output as a markdown table plus a CSV

**Done when:** we can run Claude (via CLI) and a small local Qwen (via Ollama) on the test splits and get a comparison report. Report the frontier model's pass@1 at the end of this phase. **If it's above ~85% on the test splits, stop and flag it**: the templates need to be made harder before training is worth doing.

## Phase 4: Package as a `verifiers` environment

**Goal:** the same tasks and reward, usable by Prime Intellect's eval and training tooling.

- Create `environments/ercot_sql/` as an installable `verifiers` environment module. Use `prime env init` (or the current recommended scaffold) and follow the current docs for structure.
- `load_environment(split="train", ...)` returns a single-turn environment whose dataset comes from our generated split JSONL, using our exact prompt from `env/prompt.py`.
- The rubric's reward function wraps `env/score.py`: parse SQL from the completion, execute against DuckDB, return the reward. **Do not reimplement scoring.**
- Expose the DuckDB path, split, and subset size as environment args.
- Verify it end to end with `prime eval run` (or the current equivalent) against the local Ollama/vLLM endpoint on a few tasks, and confirm rewards match our own harness on the same completions.

**Done when:** the environment installs, a `prime eval` run completes, and a test shows `verifiers` rewards and our harness rewards agree.

## Phase 5: SFT dataset (distillation)

- Run Claude via the CLI backend on the `train` split with k samples
- Keep only `correct` completions; dedupe identical SQL per task; cap examples per task
- Write a chat-format JSONL (system/user/assistant) using the exact prompt from `env/prompt.py`
- Report: tasks covered, examples per template, tasks with zero correct solutions (these are good RL targets)

Training: check whether `prime-rl` currently supports an SFT entrypoint. If it does, use it so SFT and RL share one stack. If not, use TRL or Unsloth for LoRA SFT on a Qwen3 model (size configurable, default 8B). Keep hyperparameters in a config file.

**Done when:** the dataset is built, a short SFT run completes end to end, and the resulting checkpoint is evaluated with the Phase 3 harness (served via vLLM).

## Phase 6: RL with `prime-rl`

- Install `prime-rl` following its current docs (it has CUDA requirements, so this runs on a rented GPU, not locally)
- Write a config in `configs/prime-rl/` that points the orchestrator at the `ercot_sql` environment, starting from the SFT checkpoint (or the base model if SFT was skipped)
- Train prompts: `train` tasks, weighted toward tasks the SFT model gets wrong or inconsistently right (use the Phase 3 results to build this subset)
- Log reward curves; periodically evaluate on a small fixed subset of the test splits
- **Before real training, get a smoke-test run working** (small model, a few steps) to prove the environment and reward plumbing work inside `prime-rl`
- Document the full launch command sequence in the README

**Done when:** a smoke-test run completes and logs non-constant rewards.

## ERCOT gotchas to handle correctly

These matter both for data correctness and as material for hard questions:
- ERCOT native timestamps use **hour ending** (HE1–HE24) in Central Prevailing Time. `gridstatus` may normalize to interval start. Know which you're storing and document it.
- **DST**: the fall-back day has a repeated hour (ERCOT uses a DST flag); the spring-forward day is missing one. Days are not always 24 hours.
- Real-time prices are 15-minute settlement intervals; day-ahead prices are hourly. Averaging across granularities needs care.
- Prices can be negative and can hit the offer cap. Don't filter outliers.
- Settlement point names differ between hubs and load zones; keep the naming consistent with ERCOT's.
- Prices are $/MWh; residential framing often uses ¢/kWh. Conversions should only happen when a question asks for them.

## Non-goals for v1

- Multi-turn agent loops or tool use beyond single SQL execution
- Resource-node prices, congestion/shadow prices, outage data
- Price forecasting or live trading/dispatch
- Publishing to the Prime Intellect Environments Hub (possible later)
- Any web UI (demo comes later)

## Open questions to raise with me rather than guess

- If a `gridstatus` dataset is unavailable, slow, or needs an API key, report options before working around it.
- If a template's answer is ambiguous under ERCOT conventions, flag it rather than silently picking one interpretation.
- If a Claude Code CLI flag needed to disable tools or replace the system prompt doesn't exist in the installed version, stop and report rather than running evals with an agentic Claude.
- If the current `verifiers` / `prime-rl` APIs differ meaningfully from this spec's assumptions, summarize the difference and the proposed approach before building.
