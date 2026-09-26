from ercot_bench.eval.report import build_report, pass_at_k


def test_pass_at_k():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-9
    assert abs(pass_at_k(10, 1, 5) - 0.5) < 1e-9


def test_build_report_smoke():
    rows = []
    for tid, outs in {"a": ["correct"] * 5, "b": ["correct", "wrong_answer", "sql_error", "format_error", "wrong_answer"]}.items():
        for i, o in enumerate(outs):
            rows.append({"task_id": tid, "template_id": "t", "family": "f", "difficulty": 1, "split": "s",
                         "sample_idx": i, "model": "m", "backend": "b", "outcome": o,
                         "reward": {"correct": 1, "format_error": -0.2}.get(o, 0), "latency_s": 1.0,
                         "output_tokens": 10, "cost_usd": None})
    md = build_report(rows)
    assert "pass@5" in md and "| m (b) / s |" in md
