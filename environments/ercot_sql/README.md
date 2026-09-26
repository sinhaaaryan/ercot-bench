# ercot-sql (verifiers v1 taskset)

Wraps ERCOT-Bench tasks/prompt/scoring (`ercot_bench.env.*`) as a verifiers v1 taskset. Use the
tool-less `null` harness for single-turn text-to-SQL.

```bash
cd environments/ercot_sql
uv sync
# evaluate against any OpenAI-compatible endpoint (Ollama / vLLM / mlx_lm.server)
uv run eval ercot-sql --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.split test_in_template --env.taskset.subset-size 20 \
  --model qwen3:4b --client.base-url http://localhost:11434/v1 --client.api-key-var OLLAMA_API_KEY \
  --num-rollouts 2
```

Env args (`--env.taskset.*`): `split`, `tasks_dir`, `db_path`, `subset_size`, `task.db_path`, `task.query_timeout_s`.
