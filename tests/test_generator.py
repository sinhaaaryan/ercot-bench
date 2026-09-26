import re
import random

import pytest

from ercot_bench.db import connect_readonly
from ercot_bench.tasks.generator import generate_for_template
from ercot_bench.tasks.templates import ALL_TEMPLATES, Ctx


@pytest.fixture(scope="module")
def ctx(db_path):
    con = connect_readonly(db_path)
    yield Ctx(con, None)
    con.close()


def test_template_count_and_families():
    assert 20 <= len(ALL_TEMPLATES) <= 40
    fams = {t.family for t in ALL_TEMPLATES}
    assert fams == {"basic_aggregates", "time_conventions", "spreads", "conditional", "events", "forecast_error",
                    "battery"}
    assert {t.difficulty for t in ALL_TEMPLATES} == {1, 2, 3}


@pytest.mark.parametrize("tpl", ALL_TEMPLATES, ids=lambda t: t.id)
def test_reference_answer_nonnull_and_deterministic(tpl, ctx, db_path):
    if not tpl.available(ctx):
        pytest.skip(f"missing tables {tpl.requires}")
    tasks = generate_for_template(tpl, ctx, 5, seed=7, split="t")
    assert tasks, f"{tpl.id} produced no tasks"
    for t in tasks:
        assert t.answer is not None
        assert t.answer == tpl.reference_answer(t.params, ctx)  # recompute: same answer
        # every question states its answer format explicitly
        assert re.search(r"rounded|Answer with|Answer in|YYYY-MM-DD|integer", t.question), t.question
    # fresh connection + same seed -> identical tasks
    con2 = connect_readonly(db_path)
    again = generate_for_template(tpl, Ctx(con2, None), 5, seed=7, split="t")
    con2.close()
    assert [t.model_dump() for t in tasks] == [t.model_dump() for t in again]


def test_reference_answers_score_correct_via_env(ctx, db_path):
    """A gold SQL per answer type round-trips through the real scorer (sanity of compare())."""
    from ercot_bench.env.score import score
    tpl = next(t for t in ALL_TEMPLATES if t.id == "rt_avg_day")
    task = generate_for_template(tpl, ctx, 1, seed=1, split="t")[0]
    sql = (f"SELECT round(avg(price_usd_per_mwh), 2) FROM rt_spp_15min WHERE settlement_point = '{task.params['sp']}' "
           f"AND delivery_date = DATE '{task.params['date']}'")
    assert score(f"```sql\n{sql}\n```", task, db_path).outcome == "correct"


def test_splits_deterministic_and_disjoint(cfg):
    from ercot_bench.tasks.splits import heldout_template_ids
    ids = [t.id for t in ALL_TEMPLATES]
    a = heldout_template_ids(ids, 1234, 0.2)
    assert a == heldout_template_ids(list(reversed(ids)), 1234, 0.2)
    assert 0.15 <= len(a) / len(ids) <= 0.25
