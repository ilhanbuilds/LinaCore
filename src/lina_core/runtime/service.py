import time
from collections.abc import Iterator
from datetime import UTC, datetime
from threading import BoundedSemaphore, RLock

from lina_core.core.exceptions import RuntimeNotReadyError
from lina_core.core.lifecycle import Lifecycle, LifecycleState
from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse
from lina_core.inference.streaming import StreamEvent, StreamEventType
from lina_core.models.base import ModelProvider
from lina_core.runtime.cancellation import CancellationRegistry, CancellationToken
from lina_core.runtime.health import HealthLevel, HealthStatus
from lina_core.runtime.registry import ProviderRegistry
from lina_core.runtime.resources import ResourceReport, process_memory_bytes
from lina_core.tokenizer.base import TokenizerProvider


class RuntimeService:
    def __init__(self, *, max_concurrent_requests: int = 1) -> None:
        if max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests pozitif olmalıdır.")
        self.models: ProviderRegistry[ModelProvider] = ProviderRegistry()
        self.tokenizers: ProviderRegistry[TokenizerProvider] = ProviderRegistry()
        self.lifecycle = Lifecycle()
        self.active_model_id: str | None = None
        self.active_tokenizer_id: str | None = None
        self._cancellations = CancellationRegistry()
        self._semaphore = BoundedSemaphore(max_concurrent_requests)
        self._lock = RLock()
        self._started_at: datetime | None = None
        self._generated_tokens = 0
        self._last_latency: float | None = None
        self._last_error: str | None = None

    def register_model(self, provider: ModelProvider) -> None:
        self.models.register(provider.metadata.model_id, provider)

    def register_tokenizer(self, provider: TokenizerProvider) -> None:
        self.tokenizers.register(provider.metadata.tokenizer_id, provider)

    def startup(self, model_id: str, tokenizer_id: str) -> None:
        with self._lock:
            self.lifecycle.transition(LifecycleState.STARTING)
            try:
                model = self.models.get(model_id)
                self.tokenizers.get(tokenizer_id)
                model.load()
                self.active_model_id = model_id
                self.active_tokenizer_id = tokenizer_id
                self._started_at = datetime.now(UTC)
                self.lifecycle.transition(LifecycleState.READY)
            except Exception as exc:
                self._last_error = str(exc)
                self.lifecycle.transition(LifecycleState.FAILED)
                raise

    def shutdown(self) -> None:
        with self._lock:
            self._cancellations.cancel_all("shutdown")
            if self.active_model_id:
                self.models.get(self.active_model_id).unload()
            self.lifecycle.stop()
            self.active_model_id = None
            self.active_tokenizer_id = None

    def switch_model(self, model_id: str, tokenizer_id: str) -> None:
        with self._lock:
            current = self._model()
            current.unload()
            target = self.models.get(model_id)
            self.tokenizers.get(tokenizer_id)
            try:
                target.load()
            except Exception as exc:
                self._last_error = str(exc)
                current.load()
                raise
            self.active_model_id = model_id
            self.active_tokenizer_id = tokenizer_id

    def _model(self) -> ModelProvider:
        if self.lifecycle.state is not LifecycleState.READY or not self.active_model_id:
            raise RuntimeNotReadyError("Runtime is not ready")
        return self.models.get(self.active_model_id)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        token = self._cancellations.create(request.request_id)
        if token.cancelled:
            raise RuntimeNotReadyError("İstek inference başlamadan iptal edildi.")
        started = time.perf_counter()
        if not self._semaphore.acquire(blocking=False):
            self._cancellations.finish(request.request_id)
            raise RuntimeNotReadyError("Runtime eşzamanlı istek sınırına ulaştı.")
        try:
            response = self._model().generate(request)
            self._generated_tokens += response.usage.completion_tokens
            self._last_latency = time.perf_counter() - started
            return response
        except Exception as exc:
            self._last_error = str(exc)
            raise
        finally:
            self._semaphore.release()
            self._cancellations.finish(request.request_id)

    def stream(self, request: InferenceRequest) -> Iterator[StreamEvent]:
        token = self._cancellations.create(request.request_id)
        if not self._semaphore.acquire(blocking=False):
            self._cancellations.finish(request.request_id)
            raise RuntimeNotReadyError("Runtime eşzamanlı istek sınırına ulaştı.")
        try:
            for event in self._model().stream_generate(request):
                if token.cancelled:
                    self._model().cancel(request.request_id, token.state.reason or "user")
                    yield StreamEvent(
                        StreamEventType.CANCELLED,
                        request.request_id,
                        index=event.index,
                    )
                    return
                yield event
        finally:
            self._semaphore.release()
            self._cancellations.finish(request.request_id)

    def cancellation_token(self, request_id: str) -> CancellationToken:
        return self._cancellations.create(request_id)

    def cancel(self, request_id: str | None = None, reason: str = "user") -> bool:
        if request_id is None:
            return self._cancellations.cancel_all(reason) > 0
        provider_cancelled = (
            self._model().cancel(request_id, reason) if self.active_model_id else False
        )
        return self._cancellations.cancel(request_id, reason) or provider_cancelled

    def health(self) -> HealthStatus:
        if self.lifecycle.state is not LifecycleState.READY:
            return HealthStatus(
                HealthLevel.DEGRADED,
                "Runtime başlatılmadı.",
                {
                    "runtime_state": self.lifecycle.state,
                    "active_provider": None,
                    "active_model": None,
                    "active_tokenizer": None,
                    "loaded": False,
                    "current_requests": self._cancellations.active_count,
                    "last_error": self._last_error,
                    "uptime_seconds": 0.0,
                },
            )
        provider_health = self._model().health()
        details = dict(provider_health.details)
        details.update(
            runtime_state=self.lifecycle.state,
            active_provider=self.active_model_id,
            active_model=self.active_model_id,
            active_tokenizer=self.active_tokenizer_id,
            current_requests=self._cancellations.active_count,
            last_error=self._last_error,
            uptime_seconds=(datetime.now(UTC) - self._started_at).total_seconds()
            if self._started_at
            else 0.0,
        )
        return HealthStatus(provider_health.status, provider_health.message, details)

    def resource_usage(self) -> ResourceReport:
        provider = self._model().resource_usage() if self.active_model_id else ResourceReport()
        return ResourceReport(
            process_memory_bytes(),
            provider.model_load_seconds,
            provider.model_memory_estimate,
            self._cancellations.active_count,
            self._generated_tokens,
            self._last_latency,
        )
