import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from lina_core.backends.base import BackendAvailability, InferenceBackend
from lina_core.backends.capabilities import BackendCapability, ModelFormat
from lina_core.backends.errors import BackendError, BackendUnavailableError
from lina_core.backends.llama_cpp.availability import llama_cpp_availability
from lina_core.backends.llama_cpp.config import LlamaCppBackendConfig
from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse, TokenUsage
from lina_core.inference.streaming import StreamEvent, StreamEventType
from lina_core.runtime.cancellation import CancellationRegistry
from lina_core.runtime.health import HealthLevel, HealthStatus
from lina_core.runtime.resources import ResourceReport, process_memory_bytes
from lina_core.tokenizer.base import TokenizerProvider

EngineFactory = Callable[[LlamaCppBackendConfig], Any]


class LlamaCppBackend(InferenceBackend):
    """GGUF foundation with an injectable engine for deterministic contract tests."""

    def __init__(
        self,
        config: LlamaCppBackendConfig,
        *,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.config = config
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._load_seconds: float | None = None
        self._generated_tokens = 0
        self._last_latency: float | None = None
        self._cancellations = CancellationRegistry()

    @property
    def backend_id(self) -> str:
        return "llama.cpp"

    @property
    def version(self) -> str | None:
        return self.availability().detected_version

    def availability(self) -> BackendAvailability:
        if self._engine_factory is not None:
            base = llama_cpp_availability()
            return BackendAvailability(
                True,
                base.backend_id,
                {"test-engine": "injected"},
                (),
                "test-double",
                "Inject edilmiş test engine kullanılabilir.",
                base.supported_devices,
                ("Bu availability sonucu yalnız test double içindir.",),
            )
        return llama_cpp_availability()

    @property
    def supported_formats(self) -> frozenset[ModelFormat]:
        return frozenset({ModelFormat.GGUF})

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        return frozenset(
            {
                BackendCapability.TEXT_GENERATION,
                BackendCapability.STREAMING,
                BackendCapability.CANCELLATION,
            }
        )

    def load_model(self) -> None:
        if self._engine is not None:
            return
        if not self.config.resolved_model_path.is_file():
            raise BackendError("GGUF model dosyası bulunamadı.")
        availability = self.availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason)
        if self._engine_factory is None:
            raise BackendUnavailableError(
                "Gerçek GGUF inference entegrasyonu henüz uygulanmadı; foundation hazır."
            )
        started = time.perf_counter()
        self._engine = self._engine_factory(self.config)
        self._load_seconds = time.perf_counter() - started

    def unload_model(self) -> None:
        self._cancellations.cancel_all("unload")
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()
        self._engine = None

    def create_tokenizer(self) -> TokenizerProvider:
        raise BackendError("GGUF tokenizer adapter'ı henüz uygulanmadı.")

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        engine = self._require_engine()
        started = time.perf_counter()
        prompt = request.messages[-1].content
        result = engine.generate(prompt, request.options)
        text = str(result["text"])
        prompt_tokens = int(result.get("prompt_tokens", len(prompt.split())))
        completion_tokens = int(result.get("completion_tokens", len(text.split())))
        latency = time.perf_counter() - started
        self._generated_tokens += completion_tokens
        self._last_latency = latency
        return InferenceResponse(
            request.request_id,
            self.config.model_id,
            text,
            str(result.get("finish_reason", "stop")),
            TokenUsage(prompt_tokens, completion_tokens),
            latency,
        )

    def stream_generate(self, request: InferenceRequest) -> Iterator[StreamEvent]:
        engine = self._require_engine()
        token = self._cancellations.create(request.request_id)
        yield StreamEvent(StreamEventType.STARTED, request.request_id, index=0)
        index = 0
        try:
            for piece in engine.stream(request.messages[-1].content, request.options):
                index += 1
                if token.cancelled:
                    yield StreamEvent(StreamEventType.CANCELLED, request.request_id, index=index)
                    return
                text = str(piece)
                self._generated_tokens += 1
                yield StreamEvent(StreamEventType.TOKEN, request.request_id, text, index)
            yield StreamEvent(StreamEventType.COMPLETED, request.request_id, index=index + 1)
        except Exception:
            yield StreamEvent(
                StreamEventType.ERROR,
                request.request_id,
                "GGUF test engine streaming hatası.",
                index + 1,
            )
        finally:
            self._cancellations.finish(request.request_id)

    def cancel(self, request_id: str, reason: str = "user") -> bool:
        return self._cancellations.cancel(request_id, reason)

    def health(self) -> HealthStatus:
        loaded = self._engine is not None
        return HealthStatus(
            HealthLevel.HEALTHY if loaded else HealthLevel.DEGRADED,
            "GGUF test engine yüklü." if loaded else "GGUF model yüklü değil.",
            {
                "backend": self.backend_id,
                "loaded": loaded,
                "model_path": self.config.resolved_model_path.name,
                "foundation_only": self._engine_factory is None,
                "current_requests": self._cancellations.active_count,
            },
        )

    def resource_usage(self) -> ResourceReport:
        return ResourceReport(
            process_memory_bytes(),
            self._load_seconds,
            active_requests=self._cancellations.active_count,
            generated_tokens=self._generated_tokens,
            last_inference_latency=self._last_latency,
        )

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend_id": self.backend_id,
            "model_id": self.config.model_id,
            "format": ModelFormat.GGUF,
            "context_length": self.config.context_length,
            "foundation_only": self._engine_factory is None,
        }

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise BackendError("GGUF backend yüklenmedi.")
        return self._engine
