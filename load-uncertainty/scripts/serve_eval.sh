#!/usr/bin/env bash
# Serve one model with vLLM on our port (8300) for evaluation, alone on the GPU.
#   scripts/serve_eval.sh Qwen/Qwen3-4B-Instruct-2507 base4b
#   scripts/serve_eval.sh /hackathon-new/outputs/ercot-unc-sft-4b/export/merged sft4b
# Log: outputs/serve_<tag>.log. Ready when `curl -s localhost:8300/v1/models` lists the model.
set -euo pipefail
MODEL="$1"; TAG="${2:-eval}"
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH VLLM_USE_V2_MODEL_RUNNER=0
exec /hackathon/prime-rl/.venv/bin/inference --vllm.model "$MODEL" --vllm.max-model-len 4608 \
    --vllm.gpu-memory-utilization 0.85 --server.port 8300 --backend-port 8310 \
    --output-dir /hackathon-new/outputs > "outputs/serve_${TAG}.log" 2>&1
