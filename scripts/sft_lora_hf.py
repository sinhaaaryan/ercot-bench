"""Minimal single-GPU LoRA SFT with plain transformers (no peft/accelerate), for models prime-rl can't train yet.

Used for IFM/K2-Horizon-7B: prime-rl's SFT requires a typed renderer (token/content attribution), and K2's custom
chat template has none. This script trains on a `messages` JSONL dataset (the same data prime-rl uses), with loss
only on assistant tokens (from the chat template's {% generation %} markers), then merges the adapters into the
base weights and saves a vLLM-servable model (including the model's remote-code files).

    cd prime-rl && ../ercot-bench/scripts/guarded_run.sh sftk2 30G uv run --no-sync python \
        ../ercot-bench/scripts/sft_lora_hf.py --model IFM/K2-Horizon-7B --data ../ercot-bench/data/sft_k2 \
        --out /hackathon/outputs/ercot-sft-k2-lora --epochs 2 --template-kwargs '{"reasoning_effort": "low"}'
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    active = True  # class-wide switch: False -> behave as the frozen base (used as the RL reference policy)

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        if not LoRALinear.active:
            return self.base(x)
        dt = x.dtype
        return self.base(x) + ((x @ self.lora_A.to(dt).T) @ self.lora_B.to(dt).T) * self.scale

    def merged(self) -> nn.Linear:
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scale
            self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def apply_lora(model: nn.Module, rank: int, alpha: float) -> list[tuple[str, LoRALinear]]:
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in TARGETS and isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank, alpha)
                setattr(module, child_name, lora)
                wrapped.append((f"{name}.{child_name}" if name else child_name, lora))
    return wrapped


def load_examples(data_dir: Path, tok, max_len: int, template_kwargs: dict) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rows = [json.loads(l) for f in sorted(data_dir.glob("*.jsonl")) for l in open(f) if l.strip()]
    out, skipped = [], 0
    for r in rows:
        enc = tok.apply_chat_template(r["messages"], tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                                      **template_kwargs)
        ids, mask = enc["input_ids"], enc["assistant_masks"]
        if len(ids) > max_len or sum(mask) == 0:
            skipped += 1
            continue
        ids = torch.tensor(ids)
        labels = torch.where(torch.tensor(mask, dtype=torch.bool), ids, torch.full_like(ids, -100))
        out.append((ids, labels))
    print(f"examples: {len(out)} (skipped {skipped}); assistant tokens/example ~{sum((l != -100).sum().item() for _, l in out) / max(1, len(out)):.0f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--template-kwargs", default="{}")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    data = load_examples(a.data, tok, a.max_len, json.loads(a.template_kwargs))

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.to("cuda")
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = apply_lora(model, a.rank, a.alpha)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for _, m in wrapped for p in (m.lora_A, m.lora_B)]
    print(f"loaded in {time.time() - t0:.0f}s; LoRA on {len(wrapped)} modules, "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable params; GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    total_steps = math.ceil(len(data) * a.epochs / a.grad_accum)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warmup = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    order = []
    while len(order) < total_steps * a.grad_accum:
        epoch = list(range(len(data)))
        random.shuffle(epoch)
        order += epoch
    decoder, lm_head = model.get_decoder(), model.get_output_embeddings()
    model.train()
    t0 = time.time()
    for step in range(total_steps):
        loss_sum, tok_sum = 0.0, 0
        batch = [data[i] for i in order[step * a.grad_accum:(step + 1) * a.grad_accum]]
        n_tok = sum((l != -100).sum().item() for _, l in batch)
        for ids, labels in batch:
            ids, labels = ids[None].cuda(), labels[None].cuda()
            # Run the decoder, then the LM head only on positions that predict assistant tokens: with a 250k vocab,
            # full-sequence logits (+ fp32 copy + grads) cost ~8 GB for a ~2k-token example.
            hidden = decoder(input_ids=ids).last_hidden_state[:, :-1]
            tgt = labels[:, 1:]
            sel = tgt != -100
            logits = lm_head(hidden[sel]).float()
            loss = nn.functional.cross_entropy(logits, tgt[sel], reduction="sum")
            (loss / n_tok).backward()
            loss_sum += loss.item()
            tok_sum += sel.sum().item()
            del hidden, logits, loss
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        el = time.time() - t0
        print(f"step {step + 1}/{total_steps} loss {loss_sum / max(1, tok_sum):.4f} lr {sched.get_last_lr()[0]:.2e} "
              f"elapsed {el / 60:.1f}m eta {el / (step + 1) * (total_steps - step - 1) / 60:.1f}m "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    # save adapter (PEFT-style keys) then merge and save the full model
    a.out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    adapter = {}
    for name, m in wrapped:
        adapter[f"base_model.model.{name}.lora_A.weight"] = m.lora_A.detach().to(torch.bfloat16).cpu().contiguous()
        adapter[f"base_model.model.{name}.lora_B.weight"] = m.lora_B.detach().to(torch.bfloat16).cpu().contiguous()
    (a.out / "adapter").mkdir(exist_ok=True)
    save_file(adapter, str(a.out / "adapter" / "adapter_model.safetensors"))
    (a.out / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": a.model, "r": a.rank,
         "lora_alpha": a.alpha, "lora_dropout": 0.0, "bias": "none", "target_modules": list(TARGETS)}, indent=2))

    model.gradient_checkpointing_disable()
    for name, m in wrapped:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], m.merged())
    model.config.use_cache = True
    merged = a.out / "merged"
    model.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(merged)
    src = Path(snapshot_download(a.model, local_files_only=True))
    for p in src.iterdir():  # remote code + chat template + generation config
        if p.suffix in (".py", ".jinja") or p.name == "generation_config.json":
            shutil.copyfile(p, merged / p.name)
    print(f"done: adapter -> {a.out / 'adapter'}, merged model -> {merged}")


if __name__ == "__main__":
    main()
