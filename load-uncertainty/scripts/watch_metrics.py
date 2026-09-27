"""Print one compact health line per step from a prime-rl run's metrics.jsonl (trainer + orchestrator rows).

    /hackathon/prime-rl/.venv/bin/python scripts/watch_metrics.py outputs/ercot-unc-rl-4b            # table so far
    /hackathon/prime-rl/.venv/bin/python scripts/watch_metrics.py outputs/ercot-unc-rl-4b --follow   # keep tailing

Columns (env = first train source found):
  reward     mean [p10, p90] of the pinball-skill reward (0 = numeric baseline, 1 = perfect, -1 = invalid)
  trainable  share of rollouts in groups with reward variance (1 - trainable = zero-variance groups, discarded)
  parse      parse_ok rate; cov = p10..p90 coverage (target 0.80); pin = model pinball MW; base = baseline pinball MW
  len        mean completion tokens; trunc = share cut at max_completion_tokens
  ent        policy entropy on trained tokens; mkl = mismatch KL (trainer vs vLLM logprobs); mask = IPO-masked tokens
  gn         grad norm
Eval rows (eval/<name>/...) are printed when present, with the same reward/parse/cov/pin columns.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return  # partial last line while the run is writing


def fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--follow", action="store_true")
    a = ap.parse_args()
    files = sorted((a.run_dir / "monitors" / "file").glob("metrics.jsonl*"))
    if not files:
        raise SystemExit(f"no metrics under {a.run_dir}/monitors/file yet")
    printed: set[tuple] = set()
    while True:
        steps: dict[int, dict] = defaultdict(dict)
        for f in files:
            for r in rows(f):
                if r.get("step") is not None:
                    steps[int(r["step"])].update(r)
        env = next((k.split("/")[1] for s in steps.values() for k in s
                    if k.startswith("train/") and "/all/agent/reward/mean" in k and "/agg/" not in k), None)
        for step in sorted(steps):
            m = steps[step]
            if env:
                p = f"train/{env}/all/agent"
                key = ("train", step, m.get(f"{p}/reward/mean"), m.get("optim/grad_norm"))
                if m.get(f"{p}/reward/mean") is not None and key not in printed:
                    printed.add(key)
                    print(
                        f"step {step:4d} reward {fmt(m.get(f'{p}/reward/mean'), '+.3f')} "
                        f"[{fmt(m.get(f'{p}/reward/p10'), '+.2f')},{fmt(m.get(f'{p}/reward/p90'), '+.2f')}] "
                        f"trainable {fmt(m.get(f'{p}/is_trainable/mean'), '.2f')} "
                        f"parse {fmt(m.get(f'{p}/metrics/parse_ok/mean'), '.2f')} "
                        f"cov {fmt(m.get(f'{p}/metrics/coverage/mean'), '.2f')} "
                        f"pin {fmt(m.get(f'{p}/metrics/pinball_mw/mean'), '.0f')} "
                        f"base {fmt(m.get(f'{p}/metrics/baseline_pinball_mw/mean'), '.0f')} "
                        f"len {fmt(m.get(f'{p}/num_output_tokens/mean'), '.0f')} "
                        f"trunc {fmt(m.get(f'{p}/is_truncated/mean'), '.2f')} "
                        f"ent {fmt(m.get('entropy/all/mean'), '.3f')} "
                        f"mkl {fmt(m.get('mismatch_kl/all/mean'), '.4f')} "
                        f"mask {fmt(m.get('is_masked/mean'), '.3f')} "
                        f"gn {fmt(m.get('optim/grad_norm'), '.3f')}"
                    )
            for k in sorted(k for k in m if k.startswith("eval/") and k.endswith("/all/agent/reward/mean")):
                p = k[: -len("/reward/mean")]
                key = ("eval", step, k)
                if key not in printed:
                    printed.add(key)
                    print(
                        f"EVAL {k.split('/')[1]} step {step}: reward {m[k]:+.3f} "
                        f"parse {fmt(m.get(f'{p}/metrics/parse_ok/mean'), '.3f')} "
                        f"cov {fmt(m.get(f'{p}/metrics/coverage/mean'), '.3f')} "
                        f"pin {fmt(m.get(f'{p}/metrics/pinball_mw/mean'), '.1f')} "
                        f"base {fmt(m.get(f'{p}/metrics/baseline_pinball_mw/mean'), '.1f')} "
                        f"width {fmt(m.get(f'{p}/metrics/width_mw/mean'), '.0f')}"
                    )
        if not a.follow:
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
