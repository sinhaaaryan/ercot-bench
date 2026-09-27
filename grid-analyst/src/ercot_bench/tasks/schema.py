from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AnswerType = Literal["number", "integer", "timestamp", "category", "list"]


class Tolerance(BaseModel):
    abs: float = 0.0
    rel: float = 0.0


class Task(BaseModel):
    task_id: str
    template_id: str
    family: str
    difficulty: int
    split: str = ""
    question: str
    params: dict[str, Any]
    answer: Any
    answer_type: AnswerType
    tolerance: Tolerance = Field(default_factory=Tolerance)
    # timestamp answers: the timezone the answer is expressed in ('America/Chicago' or 'UTC')
    answer_timezone: str | None = None
    # list answers: whether order matters
    ordered: bool = False
