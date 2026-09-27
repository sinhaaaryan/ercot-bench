"""Shared data paths.

The ERCOT prices / actual-load DuckDB is built by the sibling grid-analyst project (`ercot-bench ingest`):
<repo>/grid-analyst/data/ercot.duckdb. Override with ERCOT_BENCH_DUCKDB.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ERCOT_DUCKDB = Path(os.environ.get("ERCOT_BENCH_DUCKDB", REPO_ROOT / "grid-analyst" / "data" / "ercot.duckdb"))
