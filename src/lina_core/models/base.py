from abc import ABC, abstractmethod
from collections.abc import Iterator

from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse
from lina_core.inference.streaming import StreamEvent
from lina_core.models.metadata import ModelMetadata
from lina_core.runtime.health import HealthStatus
from lina_core.runtime.resources import ResourceReport


class ModelProvider(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata: ...
    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def unload(self) -> None: ...
    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...
    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResponse: ...
    @abstractmethod
    def stream_generate(self, request: InferenceRequest) -> Iterator[StreamEvent]: ...
    @abstractmethod
    def health(self) -> HealthStatus: ...

    def cancel(self, request_id: str, reason: str = "user") -> bool:
        return False

    def resource_usage(self) -> ResourceReport:
        return ResourceReport()
