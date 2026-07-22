from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("Token counts cannot be negative")


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    request_id: str
    model_id: str
    text: str
    finish_reason: str
    usage: TokenUsage
    latency: float
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.latency < 0:
            raise ValueError("Latency cannot be negative")
