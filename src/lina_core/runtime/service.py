from collections.abc import Iterator
from threading import Event, RLock
from lina_core.core.exceptions import RuntimeNotReadyError
from lina_core.core.lifecycle import Lifecycle, LifecycleState
from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse
from lina_core.inference.streaming import StreamEvent
from lina_core.models.base import ModelProvider
from lina_core.runtime.health import HealthLevel, HealthStatus
from lina_core.runtime.registry import ProviderRegistry
from lina_core.tokenizer.base import TokenizerProvider

class RuntimeService:
    def __init__(self) -> None:
        self.models: ProviderRegistry[ModelProvider] = ProviderRegistry(); self.tokenizers: ProviderRegistry[TokenizerProvider] = ProviderRegistry()
        self.lifecycle = Lifecycle(); self.active_model_id: str | None = None; self.active_tokenizer_id: str | None = None
        self._cancel = Event(); self._lock = RLock()
    def register_model(self, provider: ModelProvider) -> None: self.models.register(provider.metadata.model_id, provider)
    def register_tokenizer(self, provider: TokenizerProvider) -> None: self.tokenizers.register(provider.metadata.tokenizer_id, provider)
    def startup(self, model_id: str, tokenizer_id: str) -> None:
        with self._lock:
            self.lifecycle.transition(LifecycleState.STARTING)
            try:
                model = self.models.get(model_id); self.tokenizers.get(tokenizer_id); model.load()
                self.active_model_id=model_id; self.active_tokenizer_id=tokenizer_id; self.lifecycle.transition(LifecycleState.READY)
            except Exception:
                self.lifecycle.transition(LifecycleState.FAILED); raise
    def shutdown(self) -> None:
        with self._lock:
            if self.active_model_id: self.models.get(self.active_model_id).unload()
            self.lifecycle.stop(); self.active_model_id=None; self.active_tokenizer_id=None
    def _model(self) -> ModelProvider:
        if self.lifecycle.state is not LifecycleState.READY or not self.active_model_id: raise RuntimeNotReadyError("Runtime is not ready")
        return self.models.get(self.active_model_id)
    def infer(self, request: InferenceRequest) -> InferenceResponse: return self._model().generate(request)
    def stream(self, request: InferenceRequest) -> Iterator[StreamEvent]:
        self._cancel.clear()
        for event in self._model().stream_generate(request):
            if self._cancel.is_set(): break
            yield event
    def cancel(self) -> None: self._cancel.set()
    def health(self) -> HealthStatus:
        if self.lifecycle.state is not LifecycleState.READY: return HealthStatus(HealthLevel.DEGRADED, "runtime not started", {"state":self.lifecycle.state})
        return self._model().health()
