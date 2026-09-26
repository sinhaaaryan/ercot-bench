"""Export a prime-rl LoRA SFT/RL checkpoint to (1) a PEFT adapter and (2) a merged, vLLM-servable bf16 model.

prime-rl's tools/convert_dcp_to_bf16.py only exports full fine-tunes, so this script:
  1. rebuilds the model with prime-rl's own loader (LoRA applied from the run's resolved config), DCP-loads the
     checkpoint, and saves the adapter exactly as prime-rl broadcasts it to vLLM (adapter_model.safetensors +
     adapter_config.json) -> <out>/adapter
  2. merges W' = W + (alpha / r) * B @ A into the base model's HF safetensors shard by shard on the GPU
     (bf16 result, fp32 math) and copies config/tokenizer files -> <out>/merged

Run inside the prime-rl venv, with the GPU otherwise free (the base model is loaded once for step 1):
    cd prime-rl && uv run --no-sync python ../ercot-bench/scripts/export_lora_merged.py \
        outputs/ercot-sft-8b-lora/checkpoints/step_80 outputs/ercot-sft-8b-lora/export
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch

PRIME_RL = Path(__file__).resolve().parents[2] / "prime-rl"


def save_adapter(ckpt_dir: Path, adapter_dir: Path) -> tuple[str, float]:
    sys.path.insert(0, str(PRIME_RL / "tools"))
    import convert_dcp_to_bf16 as C  # prime-rl's exporter module: reuse its loading helpers
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    from prime_rl.trainer.lora import get_lora_state, save_lora_config
    from prime_rl.utils.weights import save_state_dict

    dcp_dir = C.resolve_dcp_dir(ckpt_dir)
    model_config, _ = C.resolve_run_configs(dcp_dir.parent)
    assert model_config.lora is not None, "checkpoint was not trained with LoRA; use tools/convert_dcp_to_bf16.py"
    C.setup_single_process_env()
    C.setup_torch_distributed()
    C.resolve_ep(model_config)
    model = C.setup_model(model_config, C.get_parallel_dims(model_config), loading_from_checkpoint_later=True)
    C.dcp_load(state_dict={"app": C.AppState(model, [], None, None)}, checkpoint_id=dcp_dir)

    sd = {}
    for k, v in get_lora_state().adapter_state_dict().items():
        if isinstance(v, DTensor):
            v = v.full_tensor()
        sd[k] = v.to("cpu", dtype=torch.bfloat16).contiguous()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_state_dict(sd, adapter_dir, save_sharded=False, adapter=True)
    save_lora_config(model, adapter_dir, rank=model_config.lora.rank, alpha=model_config.lora.alpha,
                     dropout=model_config.lora.dropout)
    base = model_config.name
    scale = model_config.lora.alpha / model_config.lora.rank
    del model
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    print(f"adapter: {len(sd)} tensors -> {adapter_dir} (base={base}, scale={scale})")
    return base, scale


def merge(base: str, adapter_dir: Path, merged_dir: Path, scale: float) -> None:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir = Path(base) if Path(base).is_dir() else Path(snapshot_download(base, local_files_only=True))
    ad = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            ad[k] = f.get_tensor(k)
    # adapter keys: [base_model.model.]<module>.lora_A.weight / .lora_B.weight
    pairs = {}
    for k in ad:
        for tag in ("lora_A", "lora_B"):
            suffix = f".{tag}.weight"
            if k.endswith(suffix):
                mod = k[: -len(suffix)].removeprefix("base_model.model.")
                pairs.setdefault(mod + ".weight", {})[tag] = ad[k]
    assert pairs and all(len(v) == 2 for v in pairs.values()), "unexpected adapter key layout"

    merged_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for shard in sorted(base_dir.glob("*.safetensors")):
        out = {}
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                w = f.get_tensor(k)
                if k in pairs:
                    a, b = pairs[k]["lora_A"].to(dev, torch.float32), pairs[k]["lora_B"].to(dev, torch.float32)
                    w = (w.to(dev, torch.float32) + scale * (b @ a)).to(torch.bfloat16).cpu()
                    done.add(k)
                out[k] = w.contiguous()
        save_file(out, str(merged_dir / shard.name), metadata={"format": "pt"})
        print(f"wrote {shard.name}")
        del out
        gc.collect()
    missing = set(pairs) - done
    assert not missing, f"{len(missing)} adapter modules not found in base weights, e.g. {sorted(missing)[:3]}"
    for p in base_dir.iterdir():
        if p.suffix != ".safetensors" and p.is_file():
            shutil.copyfile(p, merged_dir / p.name)
    (merged_dir / "ercot_merge_info.json").write_text(json.dumps(
        {"base": base, "adapter": str(adapter_dir), "scale": scale, "merged_modules": len(done)}, indent=2))
    print(f"merged {len(done)} modules -> {merged_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path, help="<run>/checkpoints/step_N")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--skip-adapter", action="store_true", help="reuse an existing <out>/adapter")
    ap.add_argument("--base", default=None, help="base model (default: from the run config)")
    ap.add_argument("--scale", type=float, default=None, help="alpha/r (default: from the run config)")
    a = ap.parse_args()
    adapter_dir = a.out_dir / "adapter"
    if a.skip_adapter:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        base = a.base or cfg["base_model_name_or_path"]
        scale = a.scale or cfg["lora_alpha"] / cfg["r"]
    else:
        base, scale = save_adapter(a.ckpt_dir, adapter_dir)
    merge(base, adapter_dir, a.out_dir / "merged", scale)


if __name__ == "__main__":
    main()
