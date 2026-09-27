"""Pre-RL check: does the policy still explore? Samples k completions per prompt at temperature 1.0 (as RL
rollouts do) and reports parse rate, distinct outputs, and the fraction of prompt groups whose rewards vary
(GRPO-style advantages are zero for groups with identical rewards, so those groups teach nothing).

    uv run python scripts/diversity_check.py --model <served name> --split val_sample200 --n 48 --k 8
"""

import argparse
import asyncio
import json
import random

import numpy as np
from openai import AsyncOpenAI

from ercot_uncertainty.eval import load_examples
from ercot_uncertainty.task import parse_percentiles, reward_percentiles


def _ik(d):
    return {int(k): v for k, v in d.items()}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:8300/v1")
    p.add_argument("--split", default="val_sample200")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256)
    a = p.parse_args()
    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY")
    exs = load_examples(a.split)
    random.Random(0).shuffle(exs)
    exs = exs[: a.n]
    sem = asyncio.Semaphore(16)

    async def group(ex):
        async with sem:
            r = await client.chat.completions.create(model=a.model, messages=ex["prompt"], n=a.k,
                                                     temperature=a.temperature, max_tokens=a.max_tokens)
        texts = [c.message.content or "" for c in r.choices]
        base, act = _ik(ex["baseline"]), _ik(ex["actual_error"])
        rewards = [reward_percentiles(t, base, act) for t in texts]
        parsed = [parse_percentiles(t, sorted(act)) for t in texts]
        return {
            "parse": np.mean([q.ok for q in parsed]),
            "distinct": len({json.dumps(q.hours, sort_keys=True) if q.ok else t for q, t in zip(parsed, texts)}),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "echoes_baseline": np.mean([q.ok and all(abs(q.hours[h][k] - base[h][k]) <= 1.5 for h in base for k in base[h])
                                        for q in parsed]),
        }

    rows = await asyncio.gather(*(group(ex) for ex in exs))
    std = np.array([r["reward_std"] for r in rows])
    print(json.dumps({
        "model": a.model, "prompts": len(rows), "k": a.k, "temperature": a.temperature,
        "parse_rate": round(float(np.mean([r["parse"] for r in rows])), 3),
        "mean_distinct_of_k": round(float(np.mean([r["distinct"] for r in rows])), 2),
        "frac_groups_with_reward_variance": round(float(np.mean(std > 1e-6)), 3),
        "mean_within_group_reward_std": round(float(std.mean()), 4),
        "mean_reward": round(float(np.mean([r["reward_mean"] for r in rows])), 4),
        "frac_samples_echoing_baseline": round(float(np.mean([r["echoes_baseline"] for r in rows])), 3),
    }, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
