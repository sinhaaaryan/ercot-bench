#!/usr/bin/env bash
# Start a demo lineup of vLLM servers (each in a memory-capped ercot-* scope) and (re)build the `ercot` tmux session.
#   scripts/demo_servers.sh qwen8b   # Qwen3-8B+SFT (:8002) + Qwen3-1.7B base (:8001) + Haiku   [default]
#   scripts/demo_servers.sh k2       # K2-Horizon-7B+SFT (:8003) + Qwen3-1.7B base (:8001) + Haiku
#   scripts/demo_servers.sh k2rl     # K2-Horizon-7B+SFT+RL (:8003) + Qwen3-1.7B base (:8001) + Haiku
#   scripts/demo_servers.sh small    # Qwen3-1.7B+SFT (:8000) + Qwen3-1.7B base (:8001) + Haiku
#   scripts/demo_servers.sh stop     # stop all demo servers (frees the GPU)
# VRAM budget: one 7-8B model at 60-68% (K2 needs more: vLLM Transformers backend) + the 1.7B base at 22% ~= 26-29 GB of 32 GB.
set -euo pipefail
LINEUP="${1:-qwen8b}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; PRL="$(cd "$REPO/../prime-rl" && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
SERVERS="ercot-sft.scope ercot-base.scope ercot-sft8b-serve.scope ercot-k2sft-serve.scope ercot-k2-serve.scope"
systemctl --user stop $SERVERS 2>/dev/null || true
[ "$LINEUP" = stop ] && { echo "stopped demo servers"; exit 0; }

serve() {  # name model port gpu_frac [extra args...]  (prime-rl's vLLM wrapper)
  local name=$1 model=$2 port=$3 frac=$4; shift 4
  (cd "$PRL" && VLLM_USE_V2_MODEL_RUNNER=0 setsid nohup "$REPO/scripts/guarded_run.sh" "$name" 14G \
    "$UV" run --no-sync inference --vllm.model "$model" --server.port "$port" --vllm.gpu-memory-utilization "$frac" \
    --vllm.max-model-len 8192 "$@" > "/hackathon/infer_${name}.log" 2>&1 &)
  wait_up "$name" "$port" "$model"
}

serve_small() {  # plain `vllm serve` with a small batch: the wrapper doesn't expose max-num-seqs, and vLLM's default
  local name=$1 model=$2 port=$3 frac=$4   # batch profiling reserves ~6 GB, too much beside a 7-8B model
  (cd "$PRL" && VLLM_USE_V2_MODEL_RUNNER=0 setsid nohup "$REPO/scripts/guarded_run.sh" "$name" 14G \
    "$UV" run --no-sync vllm serve "$model" --port "$port" --gpu-memory-utilization "$frac" --max-model-len 4096 \
    --max-num-seqs 4 --max-num-batched-tokens 4096 --enforce-eager > "/hackathon/infer_${name}.log" 2>&1 &)
  wait_up "$name" "$port" "$model"
}

wait_up() {
  local name=$1 port=$2 model=$3
  sleep 3
  until curl -sf "localhost:$port/v1/models" >/dev/null 2>&1; do
    if ! systemctl --user is-active -q "ercot-$name.scope"; then
      # the first launch of a newly trained model can fail its KV-cache check while vLLM compiles; retry once
      if [ -z "${ERCOT_DEMO_RETRY:-}" ]; then echo "server $name failed; retrying once"; ERCOT_DEMO_RETRY=1 exec "$0" "$LINEUP"; fi
      echo "server $name failed; see /hackathon/infer_${name}.log"; exit 1
    fi
    sleep 5
  done
  echo "up: $name ($model) on :$port"
}

case "$LINEUP" in
  qwen8b) serve sft8b-serve /hackathon/outputs/ercot-sft-8b-lora/export/merged 8002 0.60; FT=sft8b ;;
  k2)     serve k2sft-serve /hackathon/outputs/ercot-sft-k2-lora/merged 8003 0.64 --vllm.trust-remote-code True; FT=k2sft ;;
  k2rl)   serve k2sft-serve /hackathon/outputs/ercot-grpo-k2/merged 8003 0.64 --vllm.trust-remote-code True; FT=k2rl ;;
  small)  serve sft /hackathon/outputs/ercot-sft-1p7b/checkpoints/step_80/weights 8000 0.35; FT=sft ;;
  *) echo "unknown lineup $LINEUP"; exit 2 ;;
esac
serve_small base Qwen/Qwen3-1.7B 8001 0.29   # 4k context: ERCOT prompts are ~2.3k tokens + <=1.5k answer

R="cd $REPO && export PATH=\$HOME/.local/bin:\$PATH"
tmux kill-session -t ercot 2>/dev/null || true
tmux new-session -d -s ercot -n compare -x 200 -y 50 "$R && uv run ercot-bench ask --targets $FT,base,haiku; exec bash"
tmux new-window -t ercot -n finetuned "$R && uv run ercot-bench ask --targets $FT; exec bash"
tmux new-window -t ercot -n base  "$R && uv run ercot-bench ask --targets base; exec bash"
tmux new-window -t ercot -n haiku "$R && uv run ercot-bench ask --targets haiku; exec bash"
tmux new-window -t ercot -n status "watch -n 5 'systemctl --user list-units --type=scope --no-legend ercot-\*; echo; free -m | head -2; echo; /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv; echo; tail -3 /hackathon/mem_guard.log'"
tmux select-window -t ercot:compare
echo "demo ready: tmux attach -t ercot   (lineup: $LINEUP; compare = $FT, base, haiku)"
