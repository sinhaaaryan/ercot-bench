"""Attach each example's zone-typical baseline loss (`reward_scale`, MW) for the RL reward (task.scaled_skill).

Scale = median over TRAIN zone-days of LightGBM v2's pinball loss, per zone (train only: no val/test information).
Written in place into every example file under data/examples (SFT chat files excluded), and to reward_scale.json.
    uv run python scripts/add_reward_scale.py
"""
import json
from pathlib import Path

import numpy as np

from ercot_uncertainty.baselines.scoring import mean_quantile_loss

EX = Path(__file__).resolve().parents[1] / "data" / "examples"
losses = {}
for line in open(EX / "train.jsonl"):
    x = json.loads(line)
    hours = sorted(x["actual_error"])
    losses.setdefault(x["zone"], []).append(
        mean_quantile_loss([x["baseline"][h] for h in hours], [x["actual_error"][h] for h in hours]))
scale = {z: round(float(np.median(v)), 2) for z, v in losses.items()}
(EX / "reward_scale.json").write_text(json.dumps(scale, indent=1))
print("zone scales (MW):", scale)
n = 0
for f in sorted(EX.rglob("*.jsonl")):
    if f.parent.name.startswith("sft") or "smoke" == f.parent.name:
        continue
    rows = [json.loads(l) for l in f.open()]
    if not rows or "zone" not in rows[0] or rows[0]["zone"] not in scale:
        continue
    for r in rows:
        r["reward_scale"] = scale[r["zone"]]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n += 1
print("files updated:", n)
