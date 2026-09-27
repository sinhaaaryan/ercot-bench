"""Single-GPU GRPO (LoRA) on the ercot-sql environment with ONE copy of the model.

prime-rl's RL runs inference (vLLM) and training as separate processes, each holding the full model: for
K2-Horizon-7B that is 2 x 16.8 GB of weights, which does not fit one 32 GB GPU. This script time-shares one model:
it samples rollouts with HF `generate` (LoRA active), scores them with the SAME reward the eval harness and the
verifiers taskset use (`ercot_bench.env.score.score`), and takes a GRPO step:

  advantage_i = (r_i - mean(r_group)) / (std(r_group) + eps)      (groups with no reward variance are skipped)
  loss = mean_i [ -A_i * mean_t logp_t  +  beta * mean_t KL_k3(pi || ref)_t ]

The reference policy is the same weights with LoRA switched off (= the SFT model), so no second copy is needed.
One gradient step per batch (fully on-policy, importance ratio = 1). Logits are computed only for completion tokens.

    cd prime-rl && ../ercot-bench/scripts/guarded_run.sh grpo 30G uv run --no-sync python \
        ../ercot-bench/scripts/grpo_lora_hf.py --model /hackathon/outputs/ercot-sft-k2-lora/merged \
        --tasks ../ercot-bench/data/tasks/rl_train_k2.jsonl --out /hackathon/outputs/ercot-grpo-k2 \
        --template-kwargs '{"reasoning_effort": "low"}'
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sft_lora_hf as base  # noqa: E402  (sets PYTORCH_CUDA_ALLOC_CONF, provides LoRALinear/apply_lora)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ercot_bench.config import load_config  # noqa: E402
from ercot_bench.env.prompt import build_prompt  # noqa: E402
from ercot_bench.env.score import score  # noqa: E402
from ercot_bench.tasks.generator import load_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--beta", type=float, default=0.02, help="KL penalty to the SFT (reference) policy")
    ap.add_argument("--template-kwargs", default="{}")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    cfg = load_config()
    tasks = load_tasks(a.tasks)
    tkw = json.loads(a.template_kwargs)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    gen_cfg_eos = json.loads((Path(a.model) / "generation_config.json").read_text()).get("eos_token_id", tok.eos_token_id)
    eos_ids = set(gen_cfg_eos if isinstance(gen_cfg_eos, list) else [gen_cfg_eos])

    # prompts (identical to the eval harness), tokenized once
    prompt_ids = {}
    for t in tasks:
        if t.task_id in prompt_ids:
            continue
        system, user = build_prompt(t.question, cfg.db_path)
        text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, **tkw)
        prompt_ids[t.task_id] = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    print(f"tasks: {len(tasks)} rows, {len(prompt_ids)} unique; prompt tokens ~{sum(len(v) for v in prompt_ids.values()) // len(prompt_ids)}")

    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = base.apply_lora(model, a.rank, a.alpha)
    params = [p for _, m in wrapped for p in (m.lora_A, m.lora_B)]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    decoder, lm_head = model.get_decoder(), model.get_output_embeddings()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    print(f"LoRA r{a.rank} on {len(wrapped)} modules ({sum(p.numel() for p in params) / 1e6:.1f}M params); "
          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def completion_logps(full: torch.Tensor, n_prompt: int) -> torch.Tensor:
        hidden = decoder(input_ids=full[None]).last_hidden_state[0, n_prompt - 1:-1]
        logits = lm_head(hidden).float()
        return torch.log_softmax(logits, -1).gather(-1, full[n_prompt:, None]).squeeze(-1)

    a.out.mkdir(parents=True, exist_ok=True)
    log_f = open(a.out / "train_log.jsonl", "w")
    order = list(range(len(tasks)))
    random.shuffle(order)
    cursor = 0
    pool = cf.ThreadPoolExecutor(8)
    t_start = time.time()
    for step in range(1, a.steps + 1):
        # ---- sample prompts
        batch = []
        while len(batch) < a.prompts_per_step:
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            batch.append(tasks[order[cursor]])
            cursor += 1
        # ---- rollouts
        t0 = time.time()
        model.eval()
        LoRALinear_active(True)
        jobs = [(t, g) for t in batch for g in range(a.group)]
        completions: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(jobs), a.gen_batch):
                chunk = jobs[i:i + a.gen_batch]
                enc = tok.pad({"input_ids": [prompt_ids[t.task_id] for t, _ in chunk]}, return_tensors="pt").to("cuda")
                out = model.generate(**enc, do_sample=True, temperature=a.temperature, top_p=1.0,
                                     max_new_tokens=a.max_new_tokens, eos_token_id=list(eos_ids),
                                     pad_token_id=tok.pad_token_id, use_cache=True)
                for row in out[:, enc.input_ids.shape[1]:]:
                    ids = row.tolist()
                    cut = next((k + 1 for k, x in enumerate(ids) if x in eos_ids), len(ids))
                    completions.append(torch.tensor(ids[:cut], dtype=torch.long))
        texts = [tok.decode(c, skip_special_tokens=True) for c in completions]
        results = list(pool.map(lambda it: score(it[1], it[0][0], cfg.db_path), zip(jobs, texts)))
        rewards = torch.tensor([r.reward for r in results])
        t_gen = time.time() - t0

        # ---- advantages (per group)
        adv = torch.zeros_like(rewards)
        active_groups = 0
        for g0 in range(0, len(jobs), a.group):
            r = rewards[g0:g0 + a.group]
            if r.std() > 1e-6:
                adv[g0:g0 + a.group] = (r - r.mean()) / (r.std() + 1e-4)
                active_groups += 1

        # ---- policy gradient step (skip zero-advantage rollouts)
        t1 = time.time()
        model.train()
        idx = [i for i in range(len(jobs)) if adv[i] != 0 and len(completions[i]) > 0]
        loss_sum = kl_sum = 0.0
        if idx:
            for i in idx:
                t, _ = jobs[i]
                p_ids = prompt_ids[t.task_id].cuda()
                full = torch.cat([p_ids, completions[i].cuda()])
                with torch.no_grad():
                    LoRALinear_active(False)
                    ref = completion_logps(full, len(p_ids))
                    LoRALinear_active(True)
                logp = completion_logps(full, len(p_ids))
                delta = ref - logp
                kl = (torch.exp(delta) - delta - 1).mean()
                loss = (-(adv[i].item()) * logp.mean() + a.beta * kl) / len(idx)
                loss.backward()
                loss_sum += loss.item()
                kl_sum += kl.item() / len(idx)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
        else:
            gn = 0.0
        t_train = time.time() - t1
        outcomes = {k: sum(r.outcome == k for r in results) / len(results)
                    for k in ("correct", "wrong_answer", "sql_error", "format_error")}
        rec = {"step": step, "reward_mean": rewards.mean().item(), **outcomes, "active_groups": active_groups,
               "groups": len(batch), "trained_rollouts": len(idx), "loss": loss_sum, "kl": kl_sum, "grad_norm": gn,
               "mean_completion_tokens": sum(len(c) for c in completions) / len(completions),
               "gen_s": round(t_gen, 1), "train_s": round(t_train, 1),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
               "elapsed_min": round((time.time() - t_start) / 60, 1)}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(f"step {step}/{a.steps} reward {rec['reward_mean']:.3f} correct {outcomes['correct']:.2f} "
              f"active {active_groups}/{len(batch)} kl {kl_sum:.4f} gn {gn:.3f} tok {rec['mean_completion_tokens']:.0f} "
              f"gen {t_gen:.0f}s train {t_train:.0f}s peak {rec['peak_gib']} GiB elapsed {rec['elapsed_min']}m", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save_adapter(wrapped, a, a.out / f"adapter_step{step}")

    # ---- merge the RL adapter into the SFT weights and save a servable model
    model.gradient_checkpointing_disable()
    for name, m in wrapped:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], m.merged())
    model.config.use_cache = True
    merged = a.out / "merged"
    model.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
    tok.padding_side = "right"
    tok.save_pretrained(merged)
    for p in Path(a.model).iterdir():
        if p.suffix in (".py", ".jinja") or p.name == "generation_config.json":
            shutil.copyfile(p, merged / p.name)
    print(f"done: merged RL model -> {merged}")


def LoRALinear_active(flag: bool) -> None:
    base.LoRALinear.active = flag


def save_adapter(wrapped, a, path: Path) -> None:
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    save_file({f"base_model.model.{n}.lora_{w}.weight": getattr(m, f"lora_{w}").detach().to(torch.bfloat16).cpu().contiguous()
               for n, m in wrapped for w in ("A", "B")}, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": a.model, "r": a.rank,
         "lora_alpha": a.alpha, "lora_dropout": 0.0, "bias": "none", "target_modules": list(base.TARGETS)}, indent=2))


if __name__ == "__main__":
    main()
