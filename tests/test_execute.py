import concurrent.futures as cf

from ercot_bench.env.execute import execute_sql, validate_sql


def test_validate_rejects_non_select():
    assert validate_sql("DROP TABLE rt_spp_15min")
    assert validate_sql("SELECT 1; SELECT 2")
    assert validate_sql("INSERT INTO x VALUES (1)")
    assert validate_sql("COPY rt_spp_15min TO 'x.csv'")
    assert validate_sql("SELECT * FROM read_csv('/etc/passwd')")
    assert validate_sql("SELECT 1") is None
    assert validate_sql("WITH a AS (SELECT 1 x) SELECT x FROM a") is None


def test_execute_basic(db_path):
    r = execute_sql("SELECT count(*) FROM rt_spp_15min", db_path)
    assert r.ok and r.rows[0][0] > 0


def test_execute_errors_are_structured(db_path):
    r = execute_sql("SELECT no_such_col FROM rt_spp_15min", db_path)
    assert not r.ok and r.error_kind == "sql_error"
    r = execute_sql("DELETE FROM rt_spp_15min", db_path)
    assert not r.ok and r.error_kind == "rejected"


def test_external_access_blocked_even_if_validation_bypassed(db_path):
    # glob() is a table function that touches the filesystem; connection-level setting must block it
    from ercot_bench.db import connect_readonly
    cur = connect_readonly(db_path)
    try:
        cur.execute("SELECT * FROM read_csv('/etc/hosts')").fetchall()
        raised = False
    except Exception:
        raised = True
    assert raised


def test_timeout(db_path):
    sql = "SELECT count(*) FROM rt_spp_15min a, rt_spp_15min b WHERE a.price_usd_per_mwh + b.price_usd_per_mwh > 1e9"
    r = execute_sql(sql, db_path, timeout_s=1.0)
    assert not r.ok and r.error_kind == "timeout"


def test_row_limit(db_path):
    r = execute_sql("SELECT * FROM rt_spp_15min", db_path, max_rows=10)
    assert r.ok and len(r.rows) == 10 and r.truncated


def test_concurrent(db_path):
    sqls = [f"SELECT avg(price_usd_per_mwh) FROM rt_spp_15min WHERE hour_ending = {h}" for h in range(1, 25)] * 4
    with cf.ThreadPoolExecutor(16) as ex:
        results = list(ex.map(lambda s: execute_sql(s, db_path), sqls))
    assert all(r.ok for r in results)
    assert abs(results[0].rows[0][0] - results[24].rows[0][0]) < 1e-9
