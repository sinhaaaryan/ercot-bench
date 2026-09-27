"""CPU-only LoRA export from a prime-rl DCP checkpoint (no GPU, no torch.distributed process group).

scripts/export_lora_merged.py rebuilds the model through prime-rl's loader, which calls torch.cuda.set_device;
this reads only the LoRA tensors straight from the DCP files with dcp.load(no_dist=True), writes a PEFT-style
adapter, then reuses export_lora_merged.merge() (which runs on CPU when no GPU is visible).

    CUDA_VISIBLE_DEVICES= /hackathon/prime-rl/.venv/bin/python scripts/export_lora_cpu.py \
        outputs/ercot-unc-sft-4b/checkpoints/step_40 outputs/ercot-unc-sft-4b/export
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from export_lora_merged import merge  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path, help="<run>/checkpoints/step_N")
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()

    dcp_dir = a.ckpt_dir / "trainer"
    run_dir = a.ckpt_dir.parent.parent
    cfg_file = next((run_dir / "configs").rglob("resolved/*.json"))
    cfg = json.loads(cfg_file.read_text())
    model_cfg = cfg["model"]
    base, rank, alpha = model_cfg["name"], model_cfg["lora"]["rank"], model_cfg["lora"]["alpha"]

    md = dcp.FileSystemReader(str(dcp_dir)).read_metadata()
    keys = [k for k in md.state_dict_metadata if k.startswith("app.model.") and ".lora_" in k]
    assert keys, "no LoRA tensors in checkpoint"
    sd = {k: torch.empty(md.state_dict_metadata[k].size, dtype=md.state_dict_metadata[k].properties.dtype)
          for k in keys}
    dcp.load(sd, checkpoint_id=str(dcp_dir), no_dist=True)

    adapter, targets = {}, set()
    for k, v in sd.items():
        # app.model.model.layers.0.self_attn.q_proj.lora_A.0 -> base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        mod, tag, idx = k.removeprefix("app.model.").rsplit(".", 2)
        assert idx == "0", f"unexpected multi-adapter index in {k}"
        adapter[f"base_model.model.{mod}.{tag}.weight"] = v.to(torch.bfloat16).contiguous()
        targets.add(mod.rsplit(".", 1)[1])
    adapter_dir = a.out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(adapter, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": base, "r": rank, "lora_alpha": alpha,
        "lora_dropout": 0.0, "target_modules": sorted(targets), "bias": "none", "task_type": "CAUSAL_LM",
    }, indent=2))
    print(f"adapter: {len(adapter)} tensors ({len(adapter) // 2} modules, targets {sorted(targets)}) -> {adapter_dir}")
    merge(base, adapter_dir, a.out_dir / "merged", alpha / rank)


if __name__ == "__main__":
    main()
