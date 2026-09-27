"""Optional LLM paraphrase of questions (behind `generate --paraphrase-model`).

Only the descriptive body is reworded. The answer-format instructions (the tail starting at the first
"If tied" / "Answer" / "Give" / "Return" sentence) are kept verbatim, and a paraphrase is rejected
unless every number, date and settlement point name from the body survives unchanged.
"""

from __future__ import annotations

import asyncio
import re

from ercot_bench.models.claude_cli import ClaudeCLIClient
from ercot_bench.tasks.schema import Task

FORMAT_START = re.compile(r"(?<=[.?] )(If tied|Answer|Give|Return|List)\b")
KEY_TOKENS = re.compile(r"[A-Z]{2}_[A-Z]+|\$?\d[\d,.:\-]*")

SYSTEM = ("Reword the user's text so it asks exactly the same thing in different words. Keep every number, date, "
          "time, unit, settlement point name and technical term exactly as written. Output only the reworded text.")


def split_question(q: str) -> tuple[str, str]:
    m = FORMAT_START.search(q)
    return (q[:m.start()].rstrip(), q[m.start():]) if m else (q, "")


async def _paraphrase_one(client: ClaudeCLIClient, task: Task, sem: asyncio.Semaphore) -> Task:
    body, tail = split_question(task.question)
    if not tail:
        return task
    async with sem:
        comp = (await client.generate(SYSTEM, body, n=1))[0]
    new = comp.text.strip()
    if comp.error or not new or sorted(KEY_TOKENS.findall(new)) != sorted(KEY_TOKENS.findall(body)):
        return task
    return task.model_copy(update={"question": f"{new} {tail}", "params": {**task.params, "_paraphrased": True}})


def paraphrase_tasks(tasks: list[Task], model: str, concurrency: int = 4) -> list[Task]:
    client = ClaudeCLIClient(model=model)
    sem = asyncio.Semaphore(concurrency)

    async def main():
        return await asyncio.gather(*(_paraphrase_one(client, t, sem) for t in tasks))

    return list(asyncio.run(main()))
