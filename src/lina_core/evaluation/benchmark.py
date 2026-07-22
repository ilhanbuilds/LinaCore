from dataclasses import asdict, dataclass
from typing import Any

from lina_core.core.exceptions import EvaluationError
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.models.base import ModelProvider


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    model_id: str
    model_load_seconds: float | None
    first_token_latency: float | None
    total_latency: float
    prompt_tokens: int
    generated_tokens: int
    tokens_per_second: float
    peak_memory: int | None = None
    real_model: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_benchmark(provider: ModelProvider, prompt: str = "Benchmark") -> BenchmarkResult:
    if not provider.is_loaded or provider.metadata.model_id == "mock":
        raise EvaluationError("Benchmark yalnızca yüklenmiş gerçek model ile çalıştırılabilir.")
    response = provider.generate(InferenceRequest((ChatMessage(ChatRole.USER, prompt),)))
    rate = response.usage.completion_tokens / response.latency if response.latency else 0.0
    return BenchmarkResult(
        response.model_id,
        None,
        None,
        response.latency,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        rate,
    )
