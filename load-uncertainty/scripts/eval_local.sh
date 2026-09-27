#!/usr/bin/env bash
# Serve a local model with vLLM (port 8300), evaluate it on the given splits, stop the server.
#   scripts/eval_local.sh Qwen/Qwen3-4B-Instruct-2507 base4b test_sample400 test
set -uo pipefail
M="$1"; TAG="$2"; shift 2; cd /hackathon-new
scripts/serve_eval.sh "$M" "$TAG" &
until curl -s -m 2 localhost:8300/v1/models | grep -q "\"id\""; do sleep 5; done
for s in "$@"; do uv run python -m ercot_uncertainty.eval run --backend openai --model "$M" --split "$s" --concurrency 32 >/dev/null 2>&1; echo "evaluated $s"; done
kill $(pgrep -u ryan -f "[b]in/inference|[v]llm-router|[V]LLM::EngineCore|[v]llm::router") 2>/dev/null
echo done
