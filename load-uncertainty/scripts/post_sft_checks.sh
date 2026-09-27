#!/usr/bin/env bash
# After an SFT run: export (merge LoRA) -> serve with vLLM -> diversity check + test_sample400 eval -> stop server.
#   scripts/post_sft_checks.sh ercot-unc-sft2b-4b
set -uo pipefail
RUN="$1"; cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH HF_HOME=/srv/afoag/models
until grep -q "SFT trainer finished" outputs/$RUN/logs/latest/trainer.log 2>/dev/null && [ -d outputs/$RUN/checkpoints/step_40 ]; do sleep 10; done
sleep 15
N=$(ls outputs/$RUN/checkpoints | sed 's/step_//' | sort -n | tail -1)
/hackathon/prime-rl/.venv/bin/python scripts/export_lora_merged.py outputs/$RUN/checkpoints/step_$N outputs/$RUN/export > outputs/${RUN}_export.log 2>&1 || { echo "export failed"; exit 1; }
M=/hackathon-new/outputs/$RUN/export/merged
scripts/serve_eval.sh $M $RUN &
until curl -s -m 2 localhost:8300/v1/models | grep -q merged; do sleep 5; done
uv run python scripts/diversity_check.py --model $M --split val_sample200 --n 48 --k 8 2>/dev/null > outputs/diversity_$RUN.json
uv run python -m ercot_uncertainty.eval run --backend openai --model $M --split test_sample400 --concurrency 32 >/dev/null 2>&1
uv run python -m ercot_uncertainty.eval run --backend openai --model $M --split test --concurrency 32 >/dev/null 2>&1
kill $(pgrep -u ryan -f "[b]in/inference|[v]llm-router|[V]LLM::EngineCore|[v]llm::router") 2>/dev/null
echo "done: $(cat outputs/diversity_$RUN.json | tr -d '\n')"
