from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from lina_core.backends.capabilities import BackendCapability, ModelFormat
from lina_core.backends.device import DeviceInfo
from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse
from lina_core.inference.streaming import StreamEvent
from lina_core.runtime.health import HealthStatus
from lina_core.runtime.resources import ResourceReport
from lina_core.tokenizer.base import TokenizerProvider


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    available: bool
    backend_id: str
    installed_packages: Mapping[str, str]
    missing_packages: tuple[str, ...]
    detected_version: str | None
    reason: str
    supported_devices: tuple[DeviceInfo, ...]
    warnings: tuple[str, ...] = ()


class InferenceBackend(ABC):
    @property
    @abstractmethod
    def backend_id(self) -> str: ...
    @property
    @abstractmethod
    def version(self) -> str | None: ...
    @abstractmethod
    def availability(self) -> BackendAvailability: ...
    @property
    @abstractmethod
    def supported_formats(self) -> frozenset[ModelFormat]: ...
    @property
    @abstractmethod
    def capabilities(self) -> frozenset[BackendCapability]: ...
    @abstractmethod
    def load_model(self) -> None: ...
    @abstractmethod
    def unload_model(self) -> None: ...
    @abstractmethod
    def create_tokenizer(self) -> TokenizerProvider: ...
    @abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResponse: ...
    @abstractmethod
    def stream_generate(self, request: InferenceRequest) -> Iterator[StreamEvent]: ...
    @abstractmethod
    def cancel(self, request_id: str, reason: str = "user") -> bool: ...
    @abstractmethod
    def health(self) -> HealthStatus: ...
    @abstractmethod
    def resource_usage(self) -> ResourceReport: ...
    @property
    @abstractmethod
    def metadata(self) -> Mapping[str, Any]: ...
