"""Template registry: one module per family."""

from ercot_bench.tasks.templates import basic, battery, conditional, events, forecast_error, spreads, time_conventions
from ercot_bench.tasks.templates.base import Ctx, Template

ALL_TEMPLATES: list[Template] = [
    *basic.TEMPLATES, *time_conventions.TEMPLATES, *spreads.TEMPLATES, *conditional.TEMPLATES,
    *events.TEMPLATES, *forecast_error.TEMPLATES, *battery.TEMPLATES,
]
TEMPLATES_BY_ID = {t.id: t for t in ALL_TEMPLATES}
assert len(TEMPLATES_BY_ID) == len(ALL_TEMPLATES), "duplicate template ids"

__all__ = ["ALL_TEMPLATES", "TEMPLATES_BY_ID", "Ctx", "Template"]
