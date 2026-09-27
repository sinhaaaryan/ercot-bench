# Training runbook: prime-rl on one 32 GB GPU

Everything needed to go from "dataset exists + GPU free" to a trained, exported, evaluated Qwen3-4B LoRA.
Order: CPU checks -> RL smoke (1.7B, 10 steps) -> SFT warm start (4B) -> RL (4B) -> export -> eval.

What was verified without a GPU (2026-09-26): env unit tests, a `vf-eval` run against a mock OpenAI server
(rewards logged by the env == `task.py` on every trace), `--dry-run` of every config below (all resolve), renderer
token parity against the HF chat templates, SFT data loading, and the LoRA merge path (CPU). Nothing below has run
on the GPU yet.

## 0. Setup (nothing to install)

- **Every launch needs the prime-rl venv on PATH** (`export PATH=/hackathon/prime-rl/.venv/bin:$PATH`): the
  inference launcher spawns `vllm-router` by bare name (`FileNotFoundError: 'vllm-router'` otherwise). All commands
  below include it. Verified in the 2026-09-26 smoke run.

- We cannot write to `/hackathon/prime-rl/.venv`, and don't need to: its binaries are run directly
  (`/hackathon/prime-rl/.venv/bin/{rl,sft,inference,vf-eval}`), never `uv run` (that would try to sync the venv).
  Our code goes on `PYTHONPATH`; the launcher, env servers and harness subprocesses inherit it:

  ```bash
  export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env
  ```
  `ercot_uncertainty` only needs numpy, which that venv has. No separate venv, no CUDA builds.
- Taskset id `ercot-uncertainty-env` -> verifiers imports module `ercot_uncertainty_env` (`-` -> `_`). Do not name it
  `ercot-uncertainty` (that would import the task package itself).
- Model weights: `HF_HOME=/srv/afoag/models` (ours, writable). Qwen3-1.7B and Qwen3-4B-Instruct-2507 are
  already downloaded there. The teammate's cache in /home/aaryan is not readable by us. Qwen3-8B is not downloaded
  (16 GB; `hf download Qwen/Qwen3-8B` if ever needed).
- Ports (all off the teammate's defaults): router :8300 (orchestrator `base_url`), vLLM engine :8310,
  vLLM DP RPC 13355, rollout ZMQ 5575.
- Outputs: `/hackathon-new/outputs/<run.name>/` (gitignored).
- The null harness runs its chat loop as a `uv` script (`~/.local/bin/uv`, cached in `~/.cache/uv`), so `uv` must be
  on `PATH`.

Files:

| Path | What |
|---|---|
| `environments/ercot_uncertainty_env/` | verifiers v1 taskset: prompt = example's `prompt` messages, reward = `task.reward_percentiles`, metrics `parse_ok`, `reordered`, `coverage`, `pinball_mw`, `width_mw`, `baseline_pinball_mw` |
| `configs/prime-rl/rl_smoke_1gpu.toml` + `infer_smoke_1p7b.toml` | Qwen3-1.7B, LoRA r32, 10 steps, ckpt every 5, eval every 5 (synthetic smoke split) |
| `configs/prime-rl/sft_4b_lora_1gpu.toml` | Qwen3-4B-Instruct-2507 LoRA SFT warm start on baseline-percentile JSON |
| `configs/prime-rl/rl_4b_lora_1gpu.toml` + `infer_4b_lora.toml` | Qwen3-4B-Instruct-2507 LoRA r32/a64 RL |
| `scripts/make_smoke_examples.py` | writes SYNTHETIC `data/examples/smoke/{train,val}.jsonl` (ids `SYNTH-*`) |
| `scripts/make_sft_examples.py` | `data/examples/{train,val}.jsonl` -> `data/examples/sft/{train,validation}.jsonl` |
| `scripts/export_lora_merged.py` | LoRA checkpoint -> PEFT adapter + merged HF model (copy of the teammate's proven script) |
| `scripts/summarize_eval_traces.py` | vf-eval run -> reward / parse / coverage / pinball, and re-checks every reward against task.py |
| `scripts/watch_metrics.py` | one health line per step from a run's `metrics.jsonl` |
| `scripts/mock_openai_server.py` | fake OpenAI endpoint for CPU pipeline tests |

## 1. CPU checks (anytime)

One line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && /hackathon/prime-rl/.venv/bin/python scripts/make_smoke_examples.py && PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env /hackathon/prime-rl/.venv/bin/python -m pytest -q -p no:cacheprovider environments/ercot_uncertainty_env/tests
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env
/hackathon/prime-rl/.venv/bin/python scripts/make_smoke_examples.py
/hackathon/prime-rl/.venv/bin/python -m pytest -q -p no:cacheprovider environments/ercot_uncertainty_env/tests
```

End-to-end through verifiers' eval CLI against a fake model (terminal 1 runs the mock server). One line each:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && /hackathon/prime-rl/.venv/bin/python scripts/mock_openai_server.py --port 8799
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env MOCK_API_KEY=x /hackathon/prime-rl/.venv/bin/vf-eval ercot-uncertainty-env --env.taskset.split smoke/val --env.agent.harness.id null --env.agent.runtime.type subprocess -m mock --client.base-url http://127.0.0.1:8799/v1 --client.api-key-var MOCK_API_KEY -n 6 -r 3 --no-push --no-rich -o outputs/eval-mock && PYTHONPATH=src /hackathon/prime-rl/.venv/bin/python scripts/summarize_eval_traces.py $(ls -d outputs/eval-mock/*/ | tail -1)
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
/hackathon/prime-rl/.venv/bin/python scripts/mock_openai_server.py --port 8799 &
export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env MOCK_API_KEY=x
/hackathon/prime-rl/.venv/bin/vf-eval ercot-uncertainty-env \
    --env.taskset.split smoke/val --env.agent.harness.id null --env.agent.runtime.type subprocess \
    -m mock --client.base-url http://127.0.0.1:8799/v1 --client.api-key-var MOCK_API_KEY \
    -n 6 -r 3 --no-push --no-rich -o outputs/eval-mock
/hackathon/prime-rl/.venv/bin/python scripts/summarize_eval_traces.py "$(ls -d outputs/eval-mock/*/ | tail -1)"
```
Expected: `reward mismatches vs task.py: 0` (verified: 18 traces, mean reward -0.33, parse 0.83, the mock cycles
good/wide/tight/out-of-order/fenced/garbage answers).

Config validation without the GPU: append `--dry-run` to any `rl` / `sft` / `inference` command below.

## 2. Launch sequence (GPU)

Preflight every time:
```bash
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total --format=csv   # need ~20 GB (smoke) / ~25 GB (4B) free
free -g                                                                          # host RAM, 40 GB total, no swap
```
Optional host-RAM watchdog for our jobs (the teammate's instance only manages his own scopes; ours needs its own
pidfile/log). Then prefix launches with `/hackathon/ercot-bench/scripts/guarded_run.sh <name> <cap>` to run them
in a RAM-capped `ercot-<name>` scope (verified that `systemd-run --user --scope` works for us):
```bash
PIDFILE=/tmp/ryan_mem_guard.pid LOG=/hackathon-new/outputs/mem_guard.log /hackathon/ercot-bench/scripts/mem_guard.sh &
```

### 2a. RL smoke: Qwen3-1.7B, 10 steps, checkpoints at 5 and 10 (~5 min)

Terminal 1 (inference), one line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && VLLM_USE_V2_MODEL_RUNNER=0 PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env /hackathon/prime-rl/.venv/bin/inference @ configs/prime-rl/infer_smoke_1p7b.toml 2>&1 | tee outputs/infer_smoke.log
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env
/hackathon/prime-rl/.venv/bin/inference @ configs/prime-rl/infer_smoke_1p7b.toml \
    2>&1 | tee outputs/infer_smoke.log
```
Terminal 2 (after `curl -s localhost:8300/v1/models` answers), one line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env /hackathon/prime-rl/.venv/bin/rl @ configs/prime-rl/rl_smoke_1gpu.toml --run.name ercot-unc-smoke 2>&1 | tee outputs/rl_smoke.log
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env
/hackathon/prime-rl/.venv/bin/rl @ configs/prime-rl/rl_smoke_1gpu.toml \
    --run.name ercot-unc-smoke \
    2>&1 | tee outputs/rl_smoke.log
```
Pass: 10 `Step N` lines in `outputs/ercot-unc-smoke/logs/latest/{trainer,orchestrator}.log`, `checkpoints/step_5`
and `step_10` exist, eval rows appear in `scripts/watch_metrics.py outputs/ercot-unc-smoke`, vLLM log shows
`Loaded new LoRA adapter ... broadcasts/step_N`. Then test reload (brief's must-have) with 2 more steps from
step_10 (inference keeps running), one line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env /hackathon/prime-rl/.venv/bin/rl @ configs/prime-rl/rl_smoke_1gpu.toml --run.name ercot-unc-smoke --resume --max-steps 12 2>&1 | tee outputs/rl_smoke_resume.log
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env
/hackathon/prime-rl/.venv/bin/rl @ configs/prime-rl/rl_smoke_1gpu.toml \
    --run.name ercot-unc-smoke --resume --max-steps 12 \
    2>&1 | tee outputs/rl_smoke_resume.log
```
Pass: trainer log says it resumed from step 10 (not "Starting from scratch") and logs steps 11-12.
Stop inference (Ctrl-C terminal 1) before the SFT.

### 2b. SFT warm start: Qwen3-4B-Instruct-2507 LoRA (~15-25 min, GPU alone)

Build data, train, export. One line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && /hackathon/prime-rl/.venv/bin/python scripts/make_sft_examples.py && /hackathon/prime-rl/.venv/bin/sft @ configs/prime-rl/sft_4b_lora_1gpu.toml --run.name ercot-unc-sft-4b 2>&1 | tee outputs/sft_4b.log && CUDA_VISIBLE_DEVICES= /hackathon/prime-rl/.venv/bin/python scripts/export_lora_cpu.py outputs/ercot-unc-sft-4b/checkpoints/step_40 outputs/ercot-unc-sft-4b/export
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
/hackathon/prime-rl/.venv/bin/python scripts/make_sft_examples.py
/hackathon/prime-rl/.venv/bin/sft @ configs/prime-rl/sft_4b_lora_1gpu.toml \
    --run.name ercot-unc-sft-4b \
    2>&1 | tee outputs/sft_4b.log
CUDA_VISIBLE_DEVICES= /hackathon/prime-rl/.venv/bin/python scripts/export_lora_cpu.py \
    outputs/ercot-unc-sft-4b/checkpoints/step_40 outputs/ercot-unc-sft-4b/export
```
Before the real data exists, the same command runs on the synthetic set with
`--data.name /hackathon-new/data/examples/smoke/sft --val.data.name /hackathon-new/data/examples/smoke/sft --max-steps 3`
(dry-run verified; build it with `make_sft_examples.py --in-dir data/examples/smoke --out-dir data/examples/smoke/sft`).
Watch: `Loss` falling (it is a copy task: should go well below 0.1), validation loss every 20 steps, `Peak Mem.`
~11-12 GiB. The SFT target is the baseline itself, so the SFT model should score reward ~0 with parse_ok ~1.0: check
it with section 4 (serve `outputs/ercot-unc-sft-4b/export/merged`) before spending the RL budget.

### 2c. RL: Qwen3-4B-Instruct-2507 LoRA (100 steps, est. ~2.5 h)

Terminal 1, one line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && VLLM_USE_V2_MODEL_RUNNER=0 PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env /hackathon/prime-rl/.venv/bin/inference @ configs/prime-rl/infer_4b_lora.toml --vllm.model /hackathon-new/outputs/ercot-unc-sft-4b/export/merged 2>&1 | tee outputs/infer_4b.log
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env
/hackathon/prime-rl/.venv/bin/inference @ configs/prime-rl/infer_4b_lora.toml \
    --vllm.model /hackathon-new/outputs/ercot-unc-sft-4b/export/merged \
    2>&1 | tee outputs/infer_4b.log
```
Terminal 2, one line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env /hackathon/prime-rl/.venv/bin/rl @ configs/prime-rl/rl_4b_lora_1gpu.toml --model.name /hackathon-new/outputs/ercot-unc-sft-4b/export/merged --run.name ercot-unc-rl-4b 2>&1 | tee outputs/rl_4b.log
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env
/hackathon/prime-rl/.venv/bin/rl @ configs/prime-rl/rl_4b_lora_1gpu.toml \
    --model.name /hackathon-new/outputs/ercot-unc-sft-4b/export/merged \
    --run.name ercot-unc-rl-4b \
    2>&1 | tee outputs/rl_4b.log
```
`--model.name` must be exactly what vLLM serves (the LoRA adapter is registered under that name). Without the SFT
warm start drop both `--vllm.model` and `--model.name` (defaults are Qwen/Qwen3-4B-Instruct-2507).
Terminal 3: `/hackathon/prime-rl/.venv/bin/python scripts/watch_metrics.py outputs/ercot-unc-rl-4b --follow`.

## 3. Health metrics (what to watch, where)

Logs: `outputs/<run>/logs/latest/{orchestrator,trainer}.log`, env workers in `logs/latest/envs/*/`.
All numbers: `outputs/<run>/monitors/file/metrics.jsonl` (one JSON row per step; stats are mean/min/max/p10/p90;
`scripts/watch_metrics.py` prints the ones below). `<env>` = `ercot-unc` (train) / `ercot-unc-val` (eval).

| Signal | Where | Healthy | Trouble |
|---|---|---|---|
| Reward mean / spread | orch `Step N ... Reward x`; `train/<env>/all/agent/reward/{mean,p10,p90}` | starts ~0 after SFT (baseline = 0), trends up; p10..p90 wide enough to rank rollouts | mean pinned at -1 = unparseable output; mean falling steadily = policy drifting worse than baseline |
| Zero-variance groups | orch warning `Discarded N/M episodes ... no_signal=K`; `train/<env>/all/agent/is_trainable/mean` (= share of rollouts in groups with signal) | continuous reward => near 1.0 | low => all 8 samples identical (entropy collapse or all invalid); raise temperature / check parse |
| Parse rate | `.../metrics/parse_ok/mean`; `.../metrics/reordered/mean` | ~1.0 after SFT; reordered ~0 | falling parse_ok = format drift (look at traces in `monitors/file/traces`) |
| Calibration | `.../metrics/coverage/mean` (p10..p90 hit rate, parsed rollouts only) | -> 0.80 | >> 0.8 = intervals too wide (reward hacking toward safe hedges); << 0.8 = overconfident |
| Pinball (MW) | `.../metrics/pinball_mw/mean` vs `.../metrics/baseline_pinball_mw/mean` | model below baseline | -- |
| Entropy | trainer `Entropy`; `entropy/all/mean` | slow decline | collapse to ~0 early (with is_trainable dropping) |
| Length / truncation | `.../num_output_tokens/mean`, `.../is_truncated/mean`, orch `Truncation %` | ~60-150 tokens (5 hours x 4 numbers), truncation 0 | growing length / truncation > 0 = rambling, eats the 256 budget |
| Mismatch KL / importance ratio | trainer `Mismatch KL`; `mismatch_kl/all/mean`, `is_masked/mean` (share of tokens whose prob moved > eps=0.3 and were masked by the IPO loss), `kl_ent_ratio/mean` | ~1e-4 (teammate's 1.7B LoRA run), is_masked ~0 | > 1e-2 or is_masked > a few % = trainer/vLLM disagree (wrong renderer/model name, stale LoRA, precision) |
| Off-policy | orch `Max Off-Policy`; `off_policy/*` | 1-3 | hitting `max_off_policy_steps` (8) => episodes dropped as stale |
| Grad norm / LR | trainer `Grad. Norm`; `optim/grad_norm` | stable, below clip 1.0 | spikes to the clip every step |
| Memory | trainer `Peak Mem.`; nvidia-smi | per budget table below | -- |
| Eval | `eval/ercot-unc-val/all/agent/reward/mean` + same `metrics/*` at steps 0, 25, 50, ... (greedy) | rises with train reward | train up, eval flat/down = overfitting the train period |

Note there is no reference-model KL in prime-rl: `trainer.loss.kl_tau` penalises the squared log-ratio against the
rollout policy (a trust region). It is 0 here; 1e-3 is the "tiny" setting if the policy moves too fast.

## 4. Resume, export, serve, evaluate

Resume (same run.name; inference can keep running; add `--max-steps N` to extend): `--resume` = latest checkpoint,
`--resume.step 50` = a specific one. Checkpoints: `outputs/<run>/checkpoints/step_N/` (trainer DCP incl. frozen base
+ LoRA + optimizer, plus orchestrator state), every 25 steps, last 3 kept. One line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env /hackathon/prime-rl/.venv/bin/rl @ configs/prime-rl/rl_4b_lora_1gpu.toml --model.name /hackathon-new/outputs/ercot-unc-sft-4b/export/merged --run.name ercot-unc-rl-4b --resume 2>&1 | tee -a outputs/rl_4b.log
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env
/hackathon/prime-rl/.venv/bin/rl @ configs/prime-rl/rl_4b_lora_1gpu.toml \
    --model.name /hackathon-new/outputs/ercot-unc-sft-4b/export/merged \
    --run.name ercot-unc-rl-4b --resume \
    2>&1 | tee -a outputs/rl_4b.log
```
A fresh start under an existing run.name needs `--clean` (deletes the run dir).

Export the RL policy. Cheapest (CPU, seconds, verified on the teammate's 1.7B RL adapter): the LoRA adapter the
trainer last broadcast to vLLM is already a PEFT adapter in `outputs/<run>/broadcasts/step_N/`; merge it into the
base it was trained on (read from its `adapter_config.json`). One line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && R=outputs/ercot-unc-rl-4b && N=$(ls $R/broadcasts | sed 's/step_//' | sort -n | tail -1) && mkdir -p $R/export/adapter && cp $R/broadcasts/step_$N/adapter_* $R/export/adapter/ && CUDA_VISIBLE_DEVICES= /hackathon/prime-rl/.venv/bin/python scripts/export_lora_merged.py --skip-adapter none $R/export
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
R=outputs/ercot-unc-rl-4b
N=$(ls $R/broadcasts | sed 's/step_//' | sort -n | tail -1)
mkdir -p $R/export/adapter
cp $R/broadcasts/step_$N/adapter_* $R/export/adapter/
CUDA_VISIBLE_DEVICES= /hackathon/prime-rl/.venv/bin/python scripts/export_lora_merged.py --skip-adapter none $R/export
```
For an older checkpoint instead: `scripts/export_lora_merged.py outputs/<run>/checkpoints/step_N outputs/<run>/export_stepN`
(loads the model once, GPU otherwise free). prime-rl's own `tools/convert_dcp_to_bf16.py` refuses LoRA checkpoints.

Serve + evaluate on test (vLLM alone, so 0.85 of the GPU is fine). Terminal 1, one line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && VLLM_USE_V2_MODEL_RUNNER=0 /hackathon/prime-rl/.venv/bin/inference --vllm.model /hackathon-new/outputs/ercot-unc-rl-4b/export/merged --vllm.max-model-len 4608 --vllm.gpu-memory-utilization 0.85 --server.port 8300 --backend-port 8310 --output-dir /hackathon-new/outputs 2>&1 | tee outputs/serve_rl4b.log
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
VLLM_USE_V2_MODEL_RUNNER=0 /hackathon/prime-rl/.venv/bin/inference \
    --vllm.model /hackathon-new/outputs/ercot-unc-rl-4b/export/merged \
    --vllm.max-model-len 4608 --vllm.gpu-memory-utilization 0.85 \
    --server.port 8300 --backend-port 8310 --output-dir /hackathon-new/outputs \
    2>&1 | tee outputs/serve_rl4b.log
```
Terminal 2, one line:
```bash
cd /hackathon-new && export PATH=/hackathon/prime-rl/.venv/bin:$PATH && PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env VLLM_API_KEY=EMPTY /hackathon/prime-rl/.venv/bin/vf-eval ercot-uncertainty-env --env.taskset.split test --env.agent.harness.id null --env.agent.runtime.type subprocess -m /hackathon-new/outputs/ercot-unc-rl-4b/export/merged --client.base-url http://localhost:8300/v1 --client.api-key-var VLLM_API_KEY --sampling.temperature 0 --sampling.max-tokens 256 -r 1 --no-push --no-rich -o outputs/eval-test && PYTHONPATH=src /hackathon/prime-rl/.venv/bin/python scripts/summarize_eval_traces.py "$(ls -dt outputs/eval-test/*/ | head -1)"
```
Multi-line:
```bash
cd /hackathon-new
export PATH=/hackathon/prime-rl/.venv/bin:$PATH
export PYTHONPATH=/hackathon-new/src:/hackathon-new/environments/ercot_uncertainty_env VLLM_API_KEY=EMPTY
/hackathon/prime-rl/.venv/bin/vf-eval ercot-uncertainty-env \
    --env.taskset.split test --env.agent.harness.id null --env.agent.runtime.type subprocess \
    -m /hackathon-new/outputs/ercot-unc-rl-4b/export/merged \
    --client.base-url http://localhost:8300/v1 --client.api-key-var VLLM_API_KEY \
    --sampling.temperature 0 --sampling.max-tokens 256 -r 1 --no-push --no-rich -o outputs/eval-test
/hackathon/prime-rl/.venv/bin/python scripts/summarize_eval_traces.py "$(ls -dt outputs/eval-test/*/ | head -1)"
```
Same command with `-m`/`--vllm.model` = `Qwen/Qwen3-4B-Instruct-2507` or the SFT export gives the untrained and
warm-start rows of the leaderboard. `traces.jsonl` holds every completion for per-day analysis.

## 5. VRAM budget (one RTX 5090: 31.84 GiB visible)

Method. vLLM: its footprint is `gpu_memory_utilization x 31.84 GiB` plus CUDA-graph memory (the teammate's logs show
graphs sit on top of the budget); inside the budget: weights + non-torch/LoRA buffers + profiled peak activation, and
the rest becomes KV cache. KV per token (bf16) = 2 x layers x kv_heads x head_dim x 2 B: 112 KiB (1.7B), 144 KiB (4B, 8B);
checked against the teammate's logs (5.84 GiB -> 54,640 tokens for 1.7B; 2.62 GiB -> 19,056 for 8B). Trainer:
weights at `optimization_dtype` + LoRA (param + grad + 2 Adam moments, same dtype) + transients for one packed
`seq_len` micro-batch (full activation checkpointing with activations offloaded to CPU by default, so mostly the
bf16 logits of the sequence: 4608 x 151,936 x 2 B = 1.3 GiB) + ~0.6 GiB CUDA context/allocator slack.
Calibrated on the teammate's runs: 1.7B LoRA-r16 RL, fp32, seq 4096 -> trainer peak 8.4 GiB (6.4 weights + 0.3 LoRA
+ 1.7 transients); 8B LoRA-r32 SFT, bf16, 4x4096 tokens/micro-step -> peak 20.0 GiB (15.3 + 0.65 + 4.1); vLLM 1.7B
weights 3.26 GiB (+0.86 non-torch with LoRA), vLLM 8B weights 15.27 GiB.

| Config | vLLM (util -> footprint) | vLLM contents | Trainer | Total | Headroom |
|---|---|---|---|---|---|
| Smoke 1.7B, LoRA r32, fp32 trainer, seq 4608 | 0.30 -> 9.6 + 0.6 graphs = 10.1 GiB | 3.2 weights, ~0.9 LoRA/non-torch, ~1.2 act, **~4.2 KV (~39k tokens)** | 6.5 weights fp32 + 0.5 LoRA/Adam + ~1.9 transients + 0.6 = **~9.5 GiB** | **~19.6 GiB** | ~12 GiB |
| RL 4B, LoRA r32, bf16 trainer, seq 4608 | 0.40 -> 12.7 + 0.6 = 13.3 GiB | 7.5 weights, ~1.0 LoRA/non-torch, ~1.3 act, **~2.9 KV (~21k tokens)** | 7.5 weights bf16 + 0.5 LoRA/Adam (66M params x 8 B) + ~2.0 transients + 0.6 = **~10.6 GiB** | **~23.9 GiB** | ~8 GiB |
| RL 4B, same, util 0.50 | 15.9 + 0.6 = 16.5 GiB | KV ~6.1 GiB (~44k tokens) | ~10.6 GiB | ~27.1 GiB | ~4.7 GiB |
| RL 4B with fp32 trainer (not used) | would need <= 0.33 -> KV ~0.7 GiB (~5k tokens, one group at a time) | | 15.0 + 1.0 + 2.2 + 0.6 = ~18.8 GiB | ~30 GiB | ~1-2 GiB, rollouts starved |
| SFT 4B, LoRA r32, bf16, 2x4608 tokens/micro-step | none | | 7.5 + 0.5 + ~3.0 (8192-token logits chunk 2.3 GiB) + 0.6 = **~11.6 GiB** | ~11.6 GiB | ~20 GiB |

KV at 0.40 is enough because the 8 rollouts of a group share the prompt prefix (vLLM prefix caching): a group costs
about prompt + 8 x 256 tokens, so ~3-4 groups (24-32 rollouts) fit at once for ~3-4k-token prompts. The orchestrator
derives its initial in-flight count from KV tokens / `max_model_len`, which is why `max_model_len` = 4608 and not 8192.
If rollouts are the bottleneck (orchestrator waits on inference, not trainer), raise util to 0.45-0.50.

### Precision (why 4B uses bf16, and the catch)

fp32 weights for the 4B trainer (the default) cost 15 GiB and leave vLLM no room (table row 4), so the 4B configs use
`optimization_dtype = "bfloat16"` like the teammate's 8B SFT. prime-rl gives LoRA parameters the base layer's dtype,
so LoRA masters and Adam moments are bf16 too. With AdamW at lr 1e-5, a step on a kaiming-initialised `lora_A` entry
(|a| ~ 0.01) is below half a bf16 ulp for most entries: in a CPU simulation (32x2560 A, 20 AdamW steps) only 19% of
entries changed at lr 1e-5 (38% at 2e-5, 75% at 5e-5, 96% at 1e-4). `lora_B` starts at 0 and learns normally, so
training still works (roughly "A frozen, B trained"), but less efficiently than fp32. SFT at lr 1e-4 is unaffected.
If RL reward is flat while is_trainable is high, the first thing to try is `--trainer.optim.lr 3e-5`.

### Can Qwen3-8B RL (LoRA) fit on one 32 GB card with this setup?

Not as-is. Numbers from the teammate's 8B runs: vLLM loads 8B bf16 weights in 15.27 GiB (15.54 GiB with non-torch),
profiles 0.95 GiB activation and 0.36 GiB CUDA graphs; the 8B LoRA-r32 SFT trainer (bf16) peaked at 20.0 GiB with
16k tokens per micro-step, i.e. 15.26 weights + 0.65 LoRA/Adam + ~4.1 transients.

- Trainer at seq 4608, 1 sequence per micro-step: 15.26 + 0.65 + ~1.9 + 0.6 = **~18.4 GiB**.
- vLLM bf16 before any KV: 15.54 + ~0.6 LoRA + 0.95 + 0.36 = **~17.5 GiB**.
- Sum **~35.9 GiB > 31.84 GiB**: short by ~4 GiB before a single KV token. Two bf16 copies of an 8B model do not fit.

What it would take, cheapest first:
1. FP8 weights for inference + trimmed trainer. prime-rl exposes online quantization only as
   `--vllm.quantization fp8_per_block` (block-scaled FP8 via DeepGEMM). Weights -> ~8.8 GiB (6.95B linear params x 1 B
   + 1.24B embedding/lm_head params kept bf16), so vLLM ~11 GiB + KV. Add `--vllm.kv-cache-dtype fp8` (passes through to
   vLLM; halves KV to 72 KiB/token), `--vllm.enforce-eager true` (drops the 0.36 GiB graphs, slower decode) and
   `trainer.model.fused_lm_head_token_chunk_size = 2048` (caps the logits transient at 0.6 GiB instead of 1.3).
   Budget: trainer ~17.7 + vLLM ~11 = ~28.7 GiB, leaving ~2.5 GiB for KV+margin => util ~0.36-0.38, ~1.5-2 GiB KV
   (~20-28k fp8 tokens, ~1-2 groups in flight) and < 1 GiB spare. Risks: (a) untested here: fp8_per_block needs a
   DeepGEMM build with block-scaled FP8 for SM120 (RTX 5090), and LoRA on top of an FP8 base must work in vLLM 0.29
   on this card -- verify with a standalone `inference` launch before anything else; (b) sampler (FP8) and trainer
   (bf16) differ, so mismatch KL and IPO-masked tokens go up -- watch `mismatch_kl` / `is_masked`; (c) < 1 GiB spare on
   a WSL2 box shared with other GPU users: fragmentation or a longer-than-expected prompt means OOM mid-run;
   (d) rollouts are KV-starved, so steps are several times slower than 4B.
2. `trainer.model.fsdp_cpu_offload = true`: frozen bf16 base (15.3 GiB) lives in pinned host RAM and streams to the GPU
   each forward/backward. GPU trainer ~3-4 GiB, so vLLM can stay bf16 at ~0.55 (no mismatch from quantization).
   Risks: host RAM is 40 GB with no swap (15.3 GiB pinned + vLLM process + activation offload + env workers ~25-30 GB;
   the teammate already OOM'd this WSL VM once with ~20-27 GB of offloaded Adam state -- use guarded_run.sh);
   PCIe streaming of 15 GiB several times per micro-step makes steps minutes long; LoRA + fsdp_cpu_offload is not
   exercised by any run here.
3. Out of reach with prime-rl on this card: 4-bit (QLoRA-style) trainer base -- the trainer's `quantization` options
   are FP8/MXFP8 compute paths (DeepGEMM/torchao, Hopper/Blackwell-datacenter oriented), not memory-saving storage.

Recommendation: RL on 4B (fits with ~8 GiB headroom); use the teammate's 8B SFT/serving results for comparison.
If 8B RL is a must, first prove option 1's vLLM half alone (FP8 + LoRA loads and serves on this GPU), then run
the 10-step smoke recipe with 8B before a real run.

## 6. Gotchas

- `uv run` inside /hackathon/prime-rl would try to write the venv: always call `.venv/bin/<tool>` directly.
- Renderer: for Qwen3-4B-Instruct-2507, `enable_thinking = true` (qwen3 renderer default) matches its chat template
  at the generation prompt token for token; `false` (the teammate's 1.7B setting) prefills an empty `<think></think>`
  the instruct model never saw. For hybrid Qwen3-1.7B keep `false` (else it thinks past 256 tokens).
- prime-rl SFT with the qwen3 renderer trains the assistant turn as `<think>\n\n</think>\n\n{json}`, so the SFT'd 4B
  emits that 6-token wrapper first. Harmless (task.py's parser takes the outermost `{...}`), but it is in the budget.
- rl `--model.name` must equal the vLLM `--vllm.model` string (LoRA adapters are registered under it).
- `batch_size` counts rollouts, not prompts (64 = 8 prompts x 8).
- Zero-variance groups are dropped (`no_signal`). With the teammate's binary SQL reward that was 90% of episodes on
  step 1; our reward is continuous, so only all-invalid groups (all -1) or identical outputs should hit this.
- `data/` and `outputs/` are gitignored; the smoke data is regenerated by `scripts/make_smoke_examples.py`.
- `data/examples/sft` must be rebuilt (`make_sft_examples.py`) whenever train/val are regenerated.
- The env tests live under `environments/ercot_uncertainty_env/tests` and skip under the project's own venv (no
  verifiers there); run them with the prime-rl venv as in section 1.
