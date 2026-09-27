"""Summarize a vf-eval run of ercot-uncertainty-env and re-check every reward against task.py.

Reads <run-dir>/traces.jsonl (vf-eval output), recomputes reward_percentiles from the last
assistant message of each trace, asserts it equals the reward the env logged, and prints
mean reward (pinball skill vs baseline), parse rate, 80%-interval coverage and pinball loss (MW).

    PYTHONPATH=src /hackathon/prime-rl/.venv/bin/python scripts/summarize_eval_traces.py <run-dir or traces.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

from ercot_uncertainty.task import reward_percentiles


def last_reply(trace: dict) -> str:
    if trace.get("root_reply") is not None:
        return trace["root_reply"].strip()
    msgs = [n["message"] for n in trace["nodes"] if (n.get("message") or {}).get("role") == "assistant"]
    return (msgs[-1].get("content") or "").strip() if msgs else ""


def main() -> None:
    p = Path(sys.argv[1])
    path = p / "traces.jsonl" if p.is_dir() else p
    rows = []
    mismatches = 0
    for line in open(path):
        ep = json.loads(line)
        for t in ep["traces"]:
            d = t["task"]["data"]
            base = {int(k): v for k, v in d["baseline"].items()}
            act = {int(k): float(v) for k, v in d["actual_error"].items()}
            got = sum(r["score"] * r.get("weight", 1.0) for r in t["rewards"].values() if r)
            want = reward_percentiles(last_reply(t), base, act) if t["ok"] else None
            if want is not None and abs(got - want) > 1e-9:
                mismatches += 1
                print(f"MISMATCH {d['example_id']}: logged {got} vs task.py {want}")
            rows.append((t["ok"], got, t["metrics"]))
    ok = [r for r in rows if r[0]]
    parsed = [m for _, _, m in ok if m.get("parse_ok") == 1.0]
    print(f"traces: {len(rows)}  errored: {len(rows) - len(ok)}  reward mismatches vs task.py: {mismatches}")
    if ok:
        print(f"reward (pinball skill vs baseline) mean: {mean(r for _, r, _ in ok):+.4f}")
        print(f"parse_ok: {len(parsed) / len(ok):.3f}")
    if parsed:
        print(f"coverage p10..p90 (parsed only, nominal 0.80): {mean(m['coverage'] for m in parsed):.3f}")
        print(f"pinball MW (parsed only): {mean(m['pinball_mw'] for m in parsed):.1f}  "
              f"baseline pinball MW (same traces): {mean(m['baseline_pinball_mw'] for m in parsed):.1f}")
        print(f"interval width MW (parsed only): {mean(m['width_mw'] for m in parsed):.0f}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
