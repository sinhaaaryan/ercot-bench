#!/usr/bin/env bash
# Overnight pipeline: RL (full prompt) -> pick best checkpoint -> export -> evals (+ text-swap test)
#                     -> SFT (numbers-only) -> RL (numbers-only) -> export -> evals -> report cards.
#
# Idempotent: every finished stage writes $OUT/state/<stage>.done and is skipped on rerun, so rerunning the script
# resumes where it stopped. RL has a watchdog (no log progress for STALL_MIN minutes -> kill + resume from the last
# checkpoint) and up to 2 retries with --resume.
#
#   MODE=test scripts/overnight.sh     # mini end-to-end test: 2-step runs, 12-example evals, results in outputs/overnight_test
#   MODE=full scripts/overnight.sh     # the real thing (launch detached: setsid nohup ... &)
#
# Log: $OUT/pipeline.log   (lines: "[time] STAGE ...", "[time] ERROR ...", "[time] DONE ...")
set -u
MODE="${MODE:-full}"
ROOT=/hackathon-new
if [ "$MODE" = test ]; then
  OUT=$ROOT/outputs/overnight_test; PFX="ovt-"; RL_STEPS=2; SFT_STEPS=2; EVAL_LIMIT=12; STALL_MIN=15
  export ERCOT_RESULTS_DIR=$OUT/results
else
  OUT=$ROOT/outputs/overnight; PFX=""; RL_STEPS=${RL_STEPS:-110}; SFT_STEPS=""; EVAL_LIMIT=0; STALL_MIN=25
fi
mkdir -p "$OUT/state" "$OUT/configs" "$OUT/logs"
# single-instance lock: a second copy of the pipeline (same MODE) refuses to start
exec 9>"$OUT/.lock"
flock -n 9 || { echo "[$(date +%H:%M:%S)] ERROR another $MODE pipeline is already running; exiting" | tee -a "$OUT/pipeline.log" >&2; exit 1; }
export PATH=/hackathon/prime-rl/.venv/bin:$HOME/.local/bin:$PATH HF_HOME=/srv/afoag/models VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONPATH=$ROOT/src:$ROOT/environments/ercot_uncertainty_env
PY=/hackathon/prime-rl/.venv/bin/python
cd "$ROOT"

SFT_FULL=$ROOT/outputs/ercot-unc-sft2b-4b/export/merged          # already trained + checked (v2 anchor, full prompt)
RUN_RL_FULL=${PFX}ercot-unc-rl-4b
RUN_SFT_NUM=${PFX}ercot-unc-sft2b-4b-numeric
RUN_RL_NUM=${PFX}ercot-unc-rl-4b-numeric

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/pipeline.log" >&2; }  # stderr: stdout is reserved for return values
done_() { [ -f "$OUT/state/$1.done" ]; }
mark() { date > "$OUT/state/$1.done"; log "DONE $1"; }

gpu_used() { /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }

stop_inference() {
  local pids
  pids=$(pgrep -u "$(id -u)" -f "[b]in/inference|[v]llm-router|[V]LLM::EngineCore|[v]llm::router")
  [ -n "$pids" ] && kill $pids 2>/dev/null
  for _ in $(seq 1 30); do pgrep -u "$(id -u)" -f "[V]LLM::EngineCore" >/dev/null || break; sleep 2; done
  pids=$(pgrep -u "$(id -u)" -f "[V]LLM::EngineCore"); [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  sleep 3
}

# start_inference <model> <tag> <config>: RL-style inference server (LoRA-enabled) on :8300
start_inference() {
  local model=$1 tag=$2 cfg=${3:-configs/prime-rl/infer_4b_lora.toml}
  stop_inference
  setsid nohup /hackathon/prime-rl/.venv/bin/inference @ "$cfg" --vllm.model "$model" > "$OUT/logs/infer_$tag.log" 2>&1 &
  for _ in $(seq 1 72); do
    curl -s -m 2 localhost:8300/v1/models | grep -q "$model" && { log "inference up ($tag, GPU $(gpu_used) MiB)"; return 0; }
    grep -qE "Traceback|OutOfMemoryError" "$OUT/logs/infer_$tag.log" && break
    sleep 5
  done
  log "ERROR inference failed to start ($tag), see $OUT/logs/infer_$tag.log"; stop_inference; return 1
}

# make_config <base> <out>: in test mode shrink eval/ckpt cadence and eval size
make_config() {
  local base=$1 out=$2
  if [ "$MODE" = test ]; then
    sed -e 's/^interval = 25 /interval = 1 /' -e 's/^num_examples = 756 /num_examples = 16 /' "$base" > "$out"
  else
    cp "$base" "$out"
  fi
}

# run_rl <config> <model> <run>
run_rl() {
  local cfg=$1 model=$2 run=$3 attempt=0 flag rc
  done_ "rl_$run" && { log "skip rl_$run (done)"; return 0; }
  while [ $attempt -le 2 ]; do
    start_inference "$model" "$run" || { attempt=$((attempt + 1)); continue; }
    if ls "$ROOT/outputs/$run/checkpoints"/step_* >/dev/null 2>&1; then flag="--resume"
    elif [ -d "$ROOT/outputs/$run" ]; then flag="--clean"; else flag=""; fi
    log "STAGE rl $run attempt $attempt ${flag:-fresh} (steps $RL_STEPS)"
    setsid /hackathon/prime-rl/.venv/bin/rl @ "$cfg" --model.name "$model" --run.name "$run" --max-steps "$RL_STEPS" $flag \
      > "$OUT/logs/rl_${run}_attempt$attempt.log" 2>&1 &
    local pid=$!
    # watchdog: newest mtime among the run's logs must keep moving
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      local newest
      newest=$(find "$ROOT/outputs/$run/logs" -name "*.log" -mmin "-${STALL_MIN}" 2>/dev/null | head -1)
      if [ -z "$newest" ] && [ -d "$ROOT/outputs/$run/logs" ]; then
        log "ERROR rl $run stalled (no log progress for ${STALL_MIN} min) -> killing for resume"
        kill -TERM -- -$pid 2>/dev/null; sleep 20; kill -KILL -- -$pid 2>/dev/null
        pkill -u "$(id -u)" -f "[o]utputs/$run" 2>/dev/null
        break
      fi
    done
    wait $pid; rc=$?
    stop_inference
    if grep -q "Training finished" "$OUT/logs/rl_${run}_attempt$attempt.log" && [ -d "$ROOT/outputs/$run/checkpoints/step_$RL_STEPS" ]; then
      mark "rl_$run"; return 0
    fi
    log "ERROR rl $run attempt $attempt ended rc=$rc without finishing; tail: $(tail -3 "$OUT/logs/rl_${run}_attempt$attempt.log" | tr '\n' ' ' | cut -c1-300)"
    attempt=$((attempt + 1))
  done
  log "ERROR rl $run failed after retries"; return 1
}

# best_step <run>: step with the highest val_select mean reward (0 = the starting SFT model)
best_step() {
  $PY - "$ROOT/outputs/$1" <<'EOF'
import re, sys, glob
best = {}
for f in sorted(glob.glob(f"{sys.argv[1]}/logs/attempt_*/orchestrator.log")):
    for line in open(f, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"Evaluated \S+ \| Policy v(\d+) \|.*?\| Reward (-?[0-9.]+)", line)
        if m:
            best[int(m.group(1))] = float(m.group(2))
if not best:
    print("0 nan"); sys.exit()
step = max(best, key=lambda k: (best[k], k))
print(step, best[step], " ".join(f"v{k}:{v:+.4f}" for k, v in sorted(best.items())), file=sys.stderr)
print(step, best[step])
EOF
}

# export_best <run> <sft_model>: prints the merged model path to evaluate
export_best() {
  local run=$1 sft=$2 step rew
  read -r step rew < <(best_step "$run" 2>>"$OUT/pipeline.log")
  log "best checkpoint for $run: step $step (val_select mean reward $rew)"
  echo "$run $step $rew" > "$OUT/state/best_$run.txt"
  if [ "$step" = 0 ]; then echo "$sft"; return 0; fi
  local dst=$ROOT/outputs/$run/export_best_step$step
  if [ ! -f "$dst/merged/ercot_merge_info.json" ]; then
    $PY scripts/export_lora_merged.py "$ROOT/outputs/$run/checkpoints/step_$step" "$dst" > "$OUT/logs/export_$run.log" 2>&1 \
      || { log "ERROR export failed for $run step $step"; return 1; }
  fi
  echo "$dst/merged"
}

# eval_model <model> <tag> <split...>: serve alone on the GPU, run eval.py on each split, stop
eval_model() {
  local model=$1 tag=$2; shift 2
  done_ "eval_$tag" && { log "skip eval_$tag (done)"; return 0; }
  stop_inference
  setsid nohup scripts/serve_eval.sh "$model" "$tag" > /dev/null 2>&1 &
  for _ in $(seq 1 72); do curl -s -m 2 localhost:8300/v1/models | grep -q "$model" && break; sleep 5; done
  curl -s -m 2 localhost:8300/v1/models | grep -q "$model" || { log "ERROR eval server did not start ($tag)"; stop_inference; return 1; }
  for s in "$@"; do
    local lim=""; [ "$EVAL_LIMIT" -gt 0 ] && lim="--limit $EVAL_LIMIT"
    uv run python -m ercot_uncertainty.eval run --backend openai --model "$model" --split "$s" --concurrency 32 $lim >> "$OUT/logs/eval_$tag.log" 2>&1 \
      && log "eval $tag on $s ok" || log "ERROR eval $tag on $s failed"
  done
  stop_inference
  mark "eval_$tag"
}

# run_sft <config> <run> <data_dir_for_numeric_or_empty>
run_sft() {
  local cfg=$1 run=$2 steps=${SFT_STEPS:-40}
  done_ "sft_$run" && { log "skip sft_$run (done)"; return 0; }
  stop_inference
  local flag=""; [ -d "$ROOT/outputs/$run" ] && flag="--clean"
  log "STAGE sft $run (steps $steps)"
  local extra=""; [ -n "$SFT_STEPS" ] && extra="--max-steps $SFT_STEPS"
  /hackathon/prime-rl/.venv/bin/sft @ "$cfg" --run.name "$run" $extra $flag > "$OUT/logs/sft_$run.log" 2>&1
  for _ in $(seq 1 360); do
    grep -q "SFT trainer finished" "$ROOT/outputs/$run/logs/latest/trainer.log" 2>/dev/null && break
    grep -qE "Traceback|OutOfMemoryError" "$ROOT/outputs/$run/logs/latest/trainer.log" 2>/dev/null && break
    sleep 10
  done
  sleep 15
  local ck; ck=$(ls -d "$ROOT/outputs/$run/checkpoints"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
  [ -n "$ck" ] || { log "ERROR sft $run produced no checkpoint"; return 1; }
  $PY scripts/export_lora_merged.py "$ck" "$ROOT/outputs/$run/export" > "$OUT/logs/export_$run.log" 2>&1 || { log "ERROR sft export $run"; return 1; }
  mark "sft_$run"
}

diversity() {  # <model> <split> <tag>
  done_ "div_$3" && return 0
  scripts/serve_eval.sh "$1" "div_$3" > /dev/null 2>&1 &
  for _ in $(seq 1 72); do curl -s -m 2 localhost:8300/v1/models | grep -q "$1" && break; sleep 5; done
  local n=48; [ "$MODE" = test ] && n=4
  uv run python scripts/diversity_check.py --model "$1" --split "$2" --n $n --k 8 > "$OUT/diversity_$3.json" 2>/dev/null
  stop_inference
  log "diversity $3: $(tr -d '\n ' < "$OUT/diversity_$3.json" | cut -c1-300)"
  mark "div_$3"
}

# ------------------------------------------------------------------------------------------------------------------
log "===== pipeline start (MODE=$MODE) GPU $(gpu_used) MiB, RAM free $(free -g | awk '/Mem/{print $7}') GB"
if [ "$(gpu_used)" -gt 3000 ]; then
  log "GPU busy ($(gpu_used) MiB): waiting up to 30 min for it to free"
  for _ in $(seq 1 180); do [ "$(gpu_used)" -lt 3000 ] && break; sleep 10; done
  [ "$(gpu_used)" -lt 3000 ] || { log "ERROR GPU still busy; aborting"; exit 1; }
fi

# 1) RL on the full prompt (the declared headline)
make_config configs/prime-rl/rl_4b_lora_1gpu.toml "$OUT/configs/rl_full.toml"
run_rl "$OUT/configs/rl_full.toml" "$SFT_FULL" "$RUN_RL_FULL" || log "ERROR continuing without RL-full results"

# 2) pick best checkpoint by val_select, export, evaluate (+ text-swap control, + SFT on winter/text-swap)
if done_ "rl_$RUN_RL_FULL"; then
  M_RL=$(export_best "$RUN_RL_FULL" "$SFT_FULL") && \
    eval_model "$M_RL" "$RUN_RL_FULL" test test_sample400 val_winter data/examples/textswap/test.jsonl
fi
eval_model "$SFT_FULL" "${PFX}sft_full_extra" val_winter data/examples/textswap/test.jsonl

# 3) numbers-only ablation: SFT data -> SFT -> diversity gate -> RL -> best -> evals  (SKIP_ABLATION=1 skips it)
if [ "${SKIP_ABLATION:-0}" != 1 ]; then
if ! done_ sftdata_numeric; then
  $PY scripts/make_sft_examples.py --in-dir data/examples/numeric --out-dir data/examples/sft_jitter_numeric \
      --jitter-width-sd 0.15 --jitter-shift-sd 0.10 --seed 0 >> "$OUT/pipeline.log" 2>&1 && mark sftdata_numeric
fi
make_config configs/prime-rl/sft_4b_lora_1gpu_v2_numeric.toml "$OUT/configs/sft_numeric.toml"
if run_sft "$OUT/configs/sft_numeric.toml" "$RUN_SFT_NUM"; then
  SFT_NUM=$ROOT/outputs/$RUN_SFT_NUM/export/merged
  diversity "$SFT_NUM" numeric/val_sample200 "$RUN_SFT_NUM"
  eval_model "$SFT_NUM" "$RUN_SFT_NUM" numeric/test numeric/test_sample400 numeric/val_winter
  make_config configs/prime-rl/rl_4b_lora_1gpu_numeric.toml "$OUT/configs/rl_numeric.toml"
  if run_rl "$OUT/configs/rl_numeric.toml" "$SFT_NUM" "$RUN_RL_NUM"; then
    M_RLN=$(export_best "$RUN_RL_NUM" "$SFT_NUM") && \
      eval_model "$M_RLN" "$RUN_RL_NUM" numeric/test numeric/test_sample400 numeric/val_winter
  fi
fi
else
  log "ablation skipped (SKIP_ABLATION=1)"
fi

# 4) LightGBM rows for the winter split + report cards
$PY - "$MODE" <<'EOF' >> "$OUT/pipeline.log" 2>&1
import json, os, sys
from dataclasses import asdict
sys.path.insert(0, "/hackathon-new/src")
from ercot_uncertainty.eval import RESULTS, load_examples, score
RESULTS.mkdir(parents=True, exist_ok=True)
for split in ("val_winter",):
    for tag, key in (("lightgbm-v1", "gbm_v1_baseline"), ("lightgbm-v2", "baseline")):
        rows = [asdict(score(ex, json.dumps({"hours": [{"he": int(h), **v} for h, v in (ex["meta"][key] if key != "baseline" else ex["baseline"]).items()]}), latency_s=0.0))
                for ex in load_examples(split)]
        (RESULTS / f"{split}__baseline__{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print("lightgbm winter rows written")
EOF
R=${ERCOT_RESULTS_DIR:-$ROOT/results}
for split in test test_sample400 val_winter; do
  ref="$ROOT/results/${split}__baseline__lightgbm-v2.jsonl"; [ -f "$ref" ] || ref="$R/${split}__baseline__lightgbm-v2.jsonl"
  files=$(ls "$R"/${split}__*.jsonl "$R"/${split}-*.jsonl "$ROOT"/results/${split}__*.jsonl "$ROOT"/results/${split}-*.jsonl 2>/dev/null | sort -u | tr '\n' ' ')
  [ "$MODE" = test ] && files=$(ls "$R"/${split}__*.jsonl "$R"/${split}-*.jsonl 2>/dev/null | tr '\n' ' ')
  [ -n "$files" ] || continue
  uv run python -m ercot_uncertainty.eval_report $files --split "$split" --reference "$ref" --out "$R/report_${split}.md" \
      >> "$OUT/logs/reports.log" 2>&1 && log "report $split written" || log "ERROR report $split"
done
log "===== pipeline finished"
