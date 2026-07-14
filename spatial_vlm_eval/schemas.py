from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Sample:
    id: str
    task: str
    image_paths: list[str]
    question: str
    answer: str
    answer_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        return cls(**data)


@dataclass(frozen=True)
class Prediction:
    id: str
    task: str
    question: str
    ground_truth: str
    raw_prediction: str
    prediction: str
    correct: bool
    metric_value: float | None
    error: str | None
