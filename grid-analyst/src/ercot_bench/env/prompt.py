"""Builds the (system, user) prompt. Identical for every model and backend."""

from __future__ import annotations

from pathlib import Path

from ercot_bench.schema_doc import schema_doc

SYSTEM_PROMPT = """\
You are an expert energy-market analyst who writes DuckDB SQL over ERCOT (Texas grid) data.

You will be given a database schema and one question. Respond with exactly one DuckDB SQL query \
that computes the answer, in a single ```sql fenced code block. You cannot run the query or see \
its results; it will be executed once and its result graded automatically.

Rules:
- Exactly one statement, SELECT or WITH ... SELECT only. No DDL, no writes, no file or network access.
- The query must return exactly one row and one column containing the final answer, unless the \
question asks for a list, in which case return one column with one row per list item.
- Do the rounding, units, and formatting the question asks for inside the query.
- Timestamps: return a TIMESTAMP (or a 'YYYY-MM-DD HH:MM' string) in the timezone the question asks for.
- Dates: return a DATE (or a 'YYYY-MM-DD' string).
- Follow ERCOT conventions as documented in the schema (hour ending, DST, 15-minute vs hourly data).
- Keep any explanation brief and put it before the code block."""

USER_TEMPLATE = """\
{schema}

Question:
{question}"""


def build_prompt(question: str, db_path: Path | str) -> tuple[str, str]:
    return SYSTEM_PROMPT, USER_TEMPLATE.format(schema=schema_doc(str(db_path)), question=question)
