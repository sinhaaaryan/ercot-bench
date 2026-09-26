"""Deterministic split assignment.

- test_heldout_templates: ~20% of templates, chosen by seed; all their tasks (any date) go here.
- train: remaining templates, questions about cfg.train_years (2023-2024).
- test_in_template: remaining templates, questions about later years (2025+).
"""

from __future__ import annotations

import random

TRAIN = "train"
TEST_IN = "test_in_template"
TEST_HELDOUT = "test_heldout_templates"
SPLITS = (TRAIN, TEST_IN, TEST_HELDOUT)


def heldout_template_ids(template_ids: list[str], seed: int, fraction: float) -> set[str]:
    """Stratified by family order-independent: sort ids, shuffle with the seed, take the first k."""
    ids = sorted(template_ids)
    rng = random.Random(f"heldout-{seed}")
    rng.shuffle(ids)
    k = max(1, round(fraction * len(ids)))
    return set(ids[:k])


def split_years(train_years: list[int], all_years: list[int]) -> dict[str, set[int]]:
    last_train = max(train_years)
    return {TRAIN: set(train_years), TEST_IN: {y for y in all_years if y > last_train}}
