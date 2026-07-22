from collections.abc import Iterator, Mapping
from typing import Any

from lina_core.backends.base import BackendAvailability, InferenceBackend
from lina_core.backends.capabilities import BackendCapability, ModelFormat
from lina_core.backends.transformers.availability import transformers_availability
from lina_core.backends.transformers.config import TransformersBackendConfig
from lina_core.backends.transformers.model import TransformersModelProvider
from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse
from lina_core.inference.streaming import StreamEvent
from lina_core.runtime.health import HealthStatus
from lina_core.runtime.resources import ResourceReport
from lina_core.tokenizer.base import TokenizerProvider


class TransformersBackend(InferenceBackend):
    """Contract adapter around a local Transformers model provider."""

    def __init__(self, config: TransformersBackendConfig) -> None:
        self.provider = TransformersModelProvider(config)

    @property
    def backend_id(self) -> str:
        return "transformers"

    @property
    def version(self) -> str | None:
        return self.availability().detected_version

    def availability(self) -> BackendAvailability:
        return transformers_availability()

    @property
    def supported_formats(self) -> frozenset[ModelFormat]:
        return frozenset({ModelFormat.TRANSFORMERS, ModelFormat.SAFETENSORS, ModelFormat.PYTORCH})

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        return frozenset(
            {
                BackendCapability.TEXT_GENERATION,
                BackendCapability.CHAT,
                BackendCapability.STREAMING,
                BackendCapability.CANCELLATION,
            }
        )

    def load_model(self) -> None:
        self.provider.load()

    def unload_model(self) -> None:
        self.provider.unload()

    def create_tokenizer(self) -> TokenizerProvider:
        return self.provider.create_tokenizer()

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        return self.provider.generate(request)

    def stream_generate(self, request: InferenceRequest) -> Iterator[StreamEvent]:
        return self.provider.stream_generate(request)

    def cancel(self, request_id: str, reason: str = "user") -> bool:
        return self.provider.cancel(request_id, reason)

    def health(self) -> HealthStatus:
        return self.provider.health()

    def resource_usage(self) -> ResourceReport:
        return self.provider.resource_usage()

    @property
    def metadata(self) -> Mapping[str, Any]:
        model = self.provider.metadata
        return {
            "backend_id": self.backend_id,
            "model_id": model.model_id,
            "formats": tuple(item.value for item in self.supported_formats),
            "capabilities": tuple(item.value for item in self.capabilities),
            "local_files_only": self.provider.config.local_files_only,
            "trust_remote_code": False,
        }
