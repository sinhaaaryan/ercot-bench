"""Build SFT warm-start data: each example's prompt + an assistant turn = its baseline percentiles as JSON.

Input : data/examples/<in-split>.jsonl        (DATA_CONTRACT format)
Output: <out-dir>/train.jsonl, <out-dir>/validation.jsonl   ({"messages": [...]} rows, prime-rl SFT format)

prime-rl's SFT loads `--data.name <dir>` with datasets.load_dataset(dir, split="train"), so only
train.jsonl is trained on; validation.jsonl is for your own checks.

The target is the numeric baseline, so SFT teaches the output format and to anchor on the
baseline (reward ~0) -- RL then has to learn when to widen/tighten. The label (actual_error) is
never used here.

    /hackathon/prime-rl/.venv/bin/python scripts/make_sft_examples.py                      # real data
    /hackathon/prime-rl/.venv/bin/python scripts/make_sft_examples.py --in-dir data/examples/smoke \
        --out-dir data/examples/smoke/sft                                                   # smoke
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def target_json(row: dict, rng: random.Random | None = None, width_sd: float = 0.0, shift_sd: float = 0.0) -> str:
    """Baseline percentiles as compact JSON. With jitter (width_sd/shift_sd > 0), one random width scale
    exp(N(0, width_sd)) and one median shift N(0, shift_sd * mean half-width) are applied per example. The
    jitter is independent of the input and the label, so SFT still learns "anchor on the baseline" (the
    conditional mean is unchanged) but keeps spread around it, which RL needs to explore. Copy-exact targets
    (width_sd = shift_sd = 0) collapse the policy onto the baseline: every temperature-1 sample identical."""
    hours = [int(h) for h in row["hours"]]
    b = {int(k): v for k, v in row["baseline"].items()}
    scale, shift = 1.0, 0.0
    if rng is not None and (width_sd > 0 or shift_sd > 0):
        half = sum((b[h]["p90"] - b[h]["p10"]) / 2 for h in hours) / len(hours)
        scale = pow(2.718281828459045, rng.gauss(0.0, width_sd))
        shift = rng.gauss(0.0, shift_sd * half)
    rows = []
    for h in hours:
        m = b[h]["p50"] + shift
        rows.append({"he": h, "p10": round(m - scale * (b[h]["p50"] - b[h]["p10"])), "p50": round(m),
                     "p90": round(m + scale * (b[h]["p90"] - b[h]["p50"]))})
    return json.dumps({"hours": rows}, separators=(",", ":"))


def convert(in_path: Path, out_path: Path, rng: random.Random | None = None, width_sd: float = 0.0,
            shift_sd: float = 0.0) -> int:
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path) as f, open(out_path, "w") as g:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
            messages.append({"role": "assistant", "content": target_json(row, rng, width_sd, shift_sd)})
            g.write(json.dumps({"id": row["id"], "messages": messages}) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=ROOT / "data" / "examples")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "examples" / "sft")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--jitter-width-sd", type=float, default=0.0, help="log-sd of a per-example width scale")
    ap.add_argument("--jitter-shift-sd", type=float, default=0.0, help="sd of a per-example median shift, x mean half-width")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for split, name in ((a.train_split, "train"), (a.val_split, "validation")):
        src = a.in_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"skip {src} (missing)")
            continue
        n = convert(src, a.out_dir / f"{name}.jsonl", rng, a.jitter_width_sd, a.jitter_shift_sd)
        print(f"{src} -> {a.out_dir / f'{name}.jsonl'} ({n} rows)")


if __name__ == "__main__":
    main()
