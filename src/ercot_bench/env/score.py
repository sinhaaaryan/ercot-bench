"""THE reward function. Used by the eval harness, SFT filtering, and the verifiers rubric.

score(completion_text, task, db_path) -> ScoreResult(reward, outcome, ...)
  correct       -> 1.0
  wrong_answer  -> 0.0
  sql_error     -> 0.0   (execution error, timeout, or non-SELECT statement)
  format_error  -> -0.2  (no ```sql block, or result has the wrong shape)
"""

from __future__ import annotations

import datetime as dt
import decimal
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from ercot_bench.env.execute import ExecResult, execute_sql, jsonable
from ercot_bench.tasks.schema import Task

REWARDS = {"correct": 1.0, "wrong_answer": 0.0, "sql_error": 0.0, "format_error": -0.2}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SQL_BLOCK_RE = re.compile(r"```[ \t]*sql[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class ScoreResult(BaseModel):
    reward: float
    outcome: str
    sql: str | None = None
    value: Any = None
    expected: Any = None
    detail: str | None = None
    exec_elapsed_s: float | None = None


def extract_sql(completion: str) -> str | None:
    """Last ```sql fenced block, ignoring any <think>...</think> reasoning."""
    text = _THINK_RE.sub("", completion or "")
    # an unterminated <think> means the model never finished reasoning
    if "<think>" in text.lower():
        text = text[text.lower().rfind("</think>") + 8:] if "</think>" in text.lower() else ""
    blocks = [b.strip() for b in _SQL_BLOCK_RE.findall(text) if b.strip()]
    return blocks[-1] if blocks else None


def _result(outcome: str, **kw) -> ScoreResult:
    return ScoreResult(reward=REWARDS[outcome], outcome=outcome, **kw)


def _to_float(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float, decimal.Decimal)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, str):
        try:
            return float(v.replace(",", "").replace("$", "").strip())
        except ValueError:
            return None
    return None


def _norm_timestamp(v: Any, tz: str) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(v)
    except Exception:  # noqa: BLE001
        return None
    if ts is pd.NaT:
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(tz).tz_localize(None)
    return ts.floor("min")


def _norm_category(v: Any) -> str:
    if isinstance(v, dt.datetime):
        v = v.isoformat(sep=" ")
    elif isinstance(v, dt.date):
        v = v.isoformat()
    elif isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip().strip("'\"").casefold()


def compare(value: Any, task: Task) -> bool:
    t = task.answer_type
    exp_ = task.answer
    if t == "number":
        f = _to_float(value)
        if f is None:
            return False
        tol = max(task.tolerance.abs, task.tolerance.rel * abs(float(exp_)))
        return abs(f - float(exp_)) <= tol + 1e-9
    if t == "integer":
        f = _to_float(value)
        return f is not None and abs(f - round(f)) < 1e-9 and int(round(f)) == int(exp_)
    if t == "timestamp":
        got = _norm_timestamp(value, task.answer_timezone or "UTC")
        want = _norm_timestamp(exp_, task.answer_timezone or "UTC")
        return got is not None and got == want
    if t == "category":
        return _norm_category(value) == _norm_category(exp_)
    if t == "list":
        got = [_norm_category(x) for x in value]
        want = [_norm_category(x) for x in exp_]
        return got == want if task.ordered else (len(got) == len(set(got)) and set(got) == set(want))
    raise ValueError(f"unknown answer_type {t}")


def score_exec(sql: str, res: ExecResult, task: Task) -> ScoreResult:
    """Score an already-executed query (lets callers reuse one execution)."""
    base = dict(sql=sql, expected=task.answer, exec_elapsed_s=res.elapsed_s)
    if not res.ok:
        return _result("sql_error", detail=res.error, **base)
    ncols = len(res.columns)
    if task.answer_type == "list":
        if ncols != 1:
            return _result("format_error", detail=f"list answer must be 1 column, got {ncols}", **base)
        if res.truncated:
            return _result("format_error", detail="too many rows", **base)
        value = [r[0] for r in res.rows]
    else:
        if ncols != 1 or len(res.rows) != 1:
            return _result("format_error", detail=f"expected 1 row x 1 column, got {len(res.rows)} x {ncols}",
                           value=jsonable(res.rows[:3]), **base)
        value = res.rows[0][0]
    ok = compare(value, task)
    return _result("correct" if ok else "wrong_answer", value=jsonable(value), **base)


def score(completion: str, task: Task, db_path: Path | str, timeout_s: float = 10.0,
          max_rows: int = 1000) -> ScoreResult:
    sql = extract_sql(completion)
    if sql is None:
        return _result("format_error", detail="no ```sql fenced block found", expected=task.answer)
    res = execute_sql(sql, db_path, timeout_s=timeout_s, max_rows=max_rows)
    return score_exec(sql, res, task)
