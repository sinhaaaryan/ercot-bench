import pytest

from ercot_bench.env.score import REWARDS, compare, extract_sql, score
from ercot_bench.tasks.schema import Task, Tolerance


def T(answer, answer_type, **kw):
    return Task(task_id="t", template_id="x", family="f", difficulty=1, question="q", params={},
                answer=answer, answer_type=answer_type, **kw)


def test_extract_sql_last_block_and_think():
    text = "<think>```sql\nSELECT 0\n```</think>Here:\n```sql\nSELECT 1\n```\nand\n```SQL\nSELECT 2;\n```"
    assert extract_sql(text) == "SELECT 2;"
    assert extract_sql("no code") is None
    assert extract_sql("<think>never closed ```sql\nSELECT 1\n```") is None
    assert extract_sql("```\nSELECT 1\n```") is None  # untagged block not accepted


def test_compare_number_tolerance():
    t = T(12.34, "number", tolerance=Tolerance(abs=0.01))
    assert compare(12.345, t) and compare(12.33, t) and not compare(12.36, t)
    assert compare("12.34", t) and not compare(None, t)


def test_compare_integer_and_category_and_list():
    assert compare(96.0, T(96, "integer")) and not compare(96.5, T(96, "integer"))
    import datetime as dt
    assert compare(dt.date(2024, 1, 2), T("2024-01-02", "category"))
    assert compare(" lz_north ", T("LZ_NORTH", "category"))
    lst = T(["2024-01-02", "2024-01-05"], "list")
    assert compare([dt.date(2024, 1, 5), "2024-01-02"], lst)
    assert not compare(["2024-01-02"], lst)
    assert not compare(["2024-01-02", "2024-01-02", "2024-01-05"], lst)


def test_compare_timestamp_tz():
    import datetime as dt
    t = T("2024-07-01 15:00", "timestamp", answer_timezone="UTC")
    assert compare(dt.datetime(2024, 7, 1, 15, 0), t)
    assert compare("2024-07-01T10:00:00-05:00", t)  # aware -> converted to UTC
    assert not compare(dt.datetime(2024, 7, 1, 10, 0), t)


def test_score_outcomes(db_path):
    t = T(1, "integer")
    assert score("```sql\nSELECT 1\n```", t, db_path).outcome == "correct"
    assert score("```sql\nSELECT 2\n```", t, db_path).outcome == "wrong_answer"
    assert score("```sql\nSELECT nope FROM x\n```", t, db_path).outcome == "sql_error"
    assert score("SELECT 1", t, db_path).outcome == "format_error"
    r = score("```sql\nSELECT 1, 2\n```", t, db_path)
    assert r.outcome == "format_error" and r.reward == REWARDS["format_error"] == -0.2
    r = score("```sql\nSELECT * FROM (VALUES (1),(2))\n```", t, db_path)
    assert r.outcome == "format_error"
