import time
from collections.abc import Iterator
from dataclasses import replace

from lina_core.core.exceptions import ModelLoadError
from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse, TokenUsage
from lina_core.inference.streaming import StreamEvent, StreamEventType
from lina_core.models.base import ModelProvider
from lina_core.models.metadata import ModelCapability, ModelMetadata
from lina_core.runtime.health import HealthLevel, HealthStatus
from lina_core.tokenizer.mock import MockTokenizerProvider


class MockModelProvider(ModelProvider):
    """Deterministic echo-style provider for architecture tests; it is not an AI model."""

    def __init__(self) -> None:
        self._loaded = False
        self._tokenizer = MockTokenizerProvider()

    @property
    def metadata(self) -> ModelMetadata:
        base = ModelMetadata(
            "mock",
            "Mock Architecture Provider",
            "1",
            "deterministic-mock",
            32768,
            frozenset(
                {ModelCapability.TEXT_GENERATION, ModelCapability.CHAT, ModelCapability.STREAMING}
            ),
            "built-in",
            "MIT",
        )
        return replace(base, loaded=self._loaded)

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _text(self, request: InferenceRequest) -> str:
        if not self._loaded:
            raise ModelLoadError("Mock provider is not loaded")
        source = request.messages[-1].content.strip()
        return f"[mock] {source}"[: request.options.max_new_tokens * 4]

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        started = time.perf_counter()
        text = self._text(request)
        prompt = sum(self._tokenizer.count_tokens(m.content) for m in request.messages)
        return InferenceResponse(
            request.request_id,
            "mock",
            text,
            "stop",
            TokenUsage(prompt, self._tokenizer.count_tokens(text)),
            time.perf_counter() - started,
        )

    def stream_generate(self, request: InferenceRequest) -> Iterator[StreamEvent]:
        yield StreamEvent(StreamEventType.STARTED, request.request_id, index=0)
        for index, word in enumerate(self._text(request).split(), 1):
            yield StreamEvent(StreamEventType.TOKEN, request.request_id, word + " ", index)
        yield StreamEvent(
            StreamEventType.COMPLETED,
            request.request_id,
            index=len(self._text(request).split()) + 1,
        )

    def health(self) -> HealthStatus:
        return HealthStatus(
            HealthLevel.HEALTHY if self._loaded else HealthLevel.DEGRADED,
            "loaded" if self._loaded else "not loaded",
            {"provider": "mock"},
        )
