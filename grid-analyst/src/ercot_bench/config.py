"""Central configuration. Everything overridable via configs/ercot_bench.toml or env vars."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

LOCAL_TZ = "America/Chicago"

DEFAULT_SETTLEMENT_POINTS = [
    "HB_BUSAVG", "HB_HOUSTON", "HB_HUBAVG", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
    "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST",
]

# ERCOT posts one fuel-mix workbook per year and updates it in place. The URL paths are not
# predictable, so they live in config. Add a new year here when ERCOT posts it.
DEFAULT_FUEL_MIX_URLS = {
    2023: "https://www.ercot.com/files/docs/2023/02/07/IntGenbyFuel2023.xlsx",
    2024: "https://www.ercot.com/files/docs/2025/02/07/IntGenbyFuel2024.xlsx",
    2025: "https://www.ercot.com/files/docs/2025/02/07/IntGenbyFuel2025.xlsx",
    2026: "https://www.ercot.com/files/docs/2026/02/09/IntGenbyFuel2026-1-.xlsx",
}


@dataclass
class Config:
    data_dir: Path = REPO_ROOT / "data"
    start_date: str = "2023-01-01"
    end_date: str | None = None  # None = latest available
    settlement_points: list[str] = field(default_factory=lambda: list(DEFAULT_SETTLEMENT_POINTS))
    fuel_mix_urls: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_FUEL_MIX_URLS))
    # split config
    train_years: list[int] = field(default_factory=lambda: [2023, 2024])
    heldout_template_fraction: float = 0.2
    seed: int = 1234
    # execution
    query_timeout_s: float = 10.0
    max_rows: int = 1000

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parquet_dir(self) -> Path:
        return self.data_dir / "parquet"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ercot.duckdb"

    @property
    def tasks_dir(self) -> Path:
        return self.data_dir / "tasks"

    @property
    def start_year(self) -> int:
        return int(self.start_date[:4])


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    path = Path(path) if path else REPO_ROOT / "configs" / "ercot_bench.toml"
    if path.exists():
        raw = tomllib.loads(path.read_text())
        for k, v in raw.items():
            if k == "fuel_mix_urls":
                v = {int(y): u for y, u in v.items()}
            if k == "data_dir":
                v = (REPO_ROOT / v) if not Path(v).is_absolute() else Path(v)
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if env_dir := os.getenv("ERCOT_BENCH_DATA_DIR"):
        cfg.data_dir = Path(env_dir).expanduser().resolve()
    return cfg
