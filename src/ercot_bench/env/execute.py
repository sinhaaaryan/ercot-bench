"""Safe, concurrent SQL execution against the read-only ERCOT DuckDB database."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from ercot_bench.db import connect_readonly

ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
# Table functions / statements that could touch the filesystem or network; external access is
# also disabled at the connection level, this is defence in depth.
FORBIDDEN_FUNCS = ("read_csv", "read_parquet", "read_json", "read_text", "read_blob", "glob",
                   "parquet_scan", "csv_scan", "sniff_csv", "httpfs", "attach", "install", "load")


@dataclass
class ExecResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None  # 'rejected' | 'sql_error' | 'timeout' | 'too_many_rows'
    elapsed_s: float = 0.0
    truncated: bool = False


def validate_sql(sql: str) -> str | None:
    """Return an error message if the SQL is not a single read-only SELECT/WITH statement."""
    try:
        stmts = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except Exception as e:  # noqa: BLE001
        # sqlglot doesn't know every DuckDB construct; fall back to a conservative lexical check
        return _lexical_check(sql, parse_error=str(e))
    if len(stmts) != 1:
        return f"expected exactly one statement, got {len(stmts)}"
    root = stmts[0]
    if isinstance(root, exp.Subquery):
        root = root.this
    if not isinstance(root, ALLOWED_ROOTS):
        return f"only SELECT/WITH queries are allowed (got {type(root).__name__})"
    lowered = sql.lower()
    for f in FORBIDDEN_FUNCS:
        if f + "(" in lowered.replace(" ", ""):
            return f"function {f} is not allowed"
    return None


def _lexical_check(sql: str, parse_error: str) -> str | None:
    body = sql.strip().rstrip(";").strip()
    if ";" in body:
        return "expected exactly one statement"
    first = body.split(None, 1)[0].lower() if body else ""
    if first not in ("select", "with", "(", "from"):
        return f"only SELECT/WITH queries are allowed (parse error: {parse_error[:200]})"
    low = body.lower().replace(" ", "")
    for f in FORBIDDEN_FUNCS:
        if f + "(" in low:
            return f"function {f} is not allowed"
    return None


def execute_sql(sql: str, db_path: Path | str, timeout_s: float = 10.0, max_rows: int = 1000) -> ExecResult:
    """Execute one read-only query. Never raises; errors are returned as structured results.

    Thread-safe: each call uses its own cursor on a shared read-only connection.
    """
    t0 = time.time()
    err = validate_sql(sql)
    if err:
        return ExecResult(ok=False, error=err, error_kind="rejected", elapsed_s=time.time() - t0)
    cur = connect_readonly(db_path)
    timed_out = threading.Event()

    def _interrupt():
        timed_out.set()
        try:
            cur.interrupt()
        except Exception:  # noqa: BLE001
            pass

    timer = threading.Timer(timeout_s, _interrupt)
    timer.start()
    try:
        rel = cur.execute(sql.strip().rstrip(";"))
        cols = [d[0] for d in (rel.description or [])]
        rows = rel.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        return ExecResult(ok=True, columns=cols, rows=[tuple(r) for r in rows],
                          elapsed_s=time.time() - t0, truncated=truncated)
    except Exception as e:  # noqa: BLE001
        if timed_out.is_set():
            return ExecResult(ok=False, error=f"query exceeded {timeout_s}s timeout", error_kind="timeout",
                              elapsed_s=time.time() - t0)
        return ExecResult(ok=False, error=f"{type(e).__name__}: {str(e)[:500]}", error_kind="sql_error",
                          elapsed_s=time.time() - t0)
    finally:
        timer.cancel()
        try:
            cur.close()
        except Exception:  # noqa: BLE001
            pass


def jsonable(v: Any) -> Any:
    """Make a DuckDB result value JSON-serializable (for logging results)."""
    import datetime as dt
    import decimal

    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v
