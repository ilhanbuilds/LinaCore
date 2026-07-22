from collections.abc import Iterator
from typing import Any

import pytest

from lina_core.backends.transformers.config import TransformersBackendConfig
from lina_core.backends.transformers.model import TransformersModelProvider
from lina_core.core.exceptions import LinaCoreError
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.inference.response import InferenceResponse
from lina_core.inference.streaming import StreamEvent
from lina_core.models.metadata import ModelMetadata
from lina_core.models.mock import MockModelProvider
from lina_core.runtime.health import HealthStatus


class TransformersProviderDouble(TransformersModelProvider):
    """Lightweight Transformers adapter double used by the reusable contract suite."""

    def __init__(self) -> None:
        super().__init__(TransformersBackendConfig("test-model", model_id="transformers-double"))
        self.delegate = MockModelProvider()

    @property
    def metadata(self) -> ModelMetadata:
        base = super().metadata
        return ModelMetadata(
            "transformers-double",
            base.display_name,
            base.version,
            base.architecture,
            base.context_length,
            base.capabilities,
            "test-double",
            "test-only",
            loaded=self.delegate.is_loaded,
        )

    def load(self) -> None:
        self.delegate.load()

    def unload(self) -> None:
        self.delegate.unload()

    @property
    def is_loaded(self) -> bool:
        return self.delegate.is_loaded

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        return self.delegate.generate(request)

    def stream_generate(self, request: InferenceRequest) -> Iterator[StreamEvent]:
        return self.delegate.stream_generate(request)

    def health(self) -> HealthStatus:
        return self.delegate.health()


@pytest.fixture(
    params=[MockModelProvider, TransformersProviderDouble], ids=["mock", "transformers-double"]
)
def provider(request: pytest.FixtureRequest) -> Any:
    return request.param()


def inference_request(seed: int = 42) -> InferenceRequest:
    return InferenceRequest((ChatMessage(ChatRole.USER, "contract"),))


def test_provider_metadata(provider: Any) -> None:
    assert provider.metadata.model_id


def test_provider_initial_state(provider: Any) -> None:
    assert not provider.is_loaded


def test_provider_load(provider: Any) -> None:
    provider.load()
    assert provider.is_loaded


def test_provider_duplicate_load(provider: Any) -> None:
    provider.load()
    provider.load()
    assert provider.is_loaded


def test_provider_unload(provider: Any) -> None:
    provider.load()
    provider.unload()
    assert not provider.is_loaded


def test_provider_generate_before_load(provider: Any) -> None:
    with pytest.raises(LinaCoreError):
        provider.generate(inference_request())


def test_provider_generate(provider: Any) -> None:
    provider.load()
    assert provider.generate(inference_request()).text


def test_provider_deterministic(provider: Any) -> None:
    provider.load()
    assert (
        provider.generate(inference_request()).text == provider.generate(inference_request()).text
    )


def test_provider_streaming(provider: Any) -> None:
    provider.load()
    assert len(list(provider.stream_generate(inference_request()))) >= 2


def test_provider_terminal_event(provider: Any) -> None:
    provider.load()
    assert list(provider.stream_generate(inference_request()))[-1].type == "completed"


def test_provider_health(provider: Any) -> None:
    assert provider.health().status


def test_provider_resource_cleanup(provider: Any) -> None:
    provider.load()
    provider.unload()
    assert not provider.is_loaded


def test_provider_failure_recovery(provider: Any) -> None:
    with pytest.raises(LinaCoreError):
        provider.generate(inference_request())
    provider.load()
    assert provider.generate(inference_request()).text
