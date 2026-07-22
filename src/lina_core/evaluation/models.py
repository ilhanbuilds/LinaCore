from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from lina_core.inference.request import ChatMessage


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    sample_id: str
    category: str
    language: str
    prompt: str | None = None
    messages: tuple[ChatMessage, ...] | None = None
    expected_output: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or bool(self.prompt) == bool(self.messages):
            raise ValueError("Provide exactly one of prompt or messages")


@dataclass(frozen=True, slots=True)
class MetricResult:
    metric_name: str
    value: float
    details: Mapping[str, Any] = field(default_factory=dict)
    passed: bool | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    run_id: str
    model_id: str
    dataset_id: str
    metrics: tuple[MetricResult, ...]
    duration: float
    sample_count: int
    failures: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
