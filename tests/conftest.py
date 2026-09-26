import pytest

from ercot_bench.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def db_path(cfg):
    if not cfg.db_path.exists():
        pytest.skip("data/ercot.duckdb not built; run `ercot-bench ingest`")
    return cfg.db_path
