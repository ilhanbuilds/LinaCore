from pathlib import Path
from typing import Any

import pytest

from lina_core.backends.capabilities import BackendCapability, ModelFormat
from lina_core.backends.errors import BackendError, BackendUnavailableError
from lina_core.backends.llama_cpp.availability import llama_cpp_availability
from lina_core.backends.llama_cpp.backend import LlamaCppBackend
from lina_core.backends.llama_cpp.config import LlamaCppBackendConfig
from lina_core.core.exceptions import SecurityError
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.inference.streaming import StreamEventType


class FakeGgufEngine:
    def __init__(self) -> None:
        self.closed = False

    def generate(self, prompt: str, options: Any) -> dict[str, Any]:
        return {
            "text": f"fake:{prompt}",
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    def stream(self, prompt: str, options: Any) -> list[str]:
        del prompt, options
        return ["fake", " stream"]

    def close(self) -> None:
        self.closed = True


def config(tmp_path: Path, *, create: bool = True) -> LlamaCppBackendConfig:
    path = tmp_path / "model.gguf"
    if create:
        path.write_bytes(b"GGUF-test-double")
    return LlamaCppBackendConfig(Path("model.gguf"), tmp_path, model_id="gguf-test")


def request() -> InferenceRequest:
    return InferenceRequest((ChatMessage(ChatRole.USER, "hello gguf"),), request_id="gguf-request")


def test_llama_availability_is_typed() -> None:
    result = llama_cpp_availability()
    assert result.backend_id == "llama.cpp"
    assert isinstance(result.available, bool)
    assert result.warnings


@pytest.mark.parametrize("suffix", [".bin", ".safetensors", ".txt", ""])
def test_llama_config_requires_gguf(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(BackendError):
        LlamaCppBackendConfig(Path(f"model{suffix}"), tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_length": 0},
        {"threads": 0},
        {"gpu_layers": -1},
        {"model_id": ""},
    ],
)
def test_llama_config_rejects_invalid_values(tmp_path: Path, kwargs: dict[str, object]) -> None:
    with pytest.raises(BackendError):
        LlamaCppBackendConfig(Path("model.gguf"), tmp_path, **kwargs)  # type: ignore[arg-type]


def test_llama_config_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(SecurityError):
        LlamaCppBackendConfig(Path("../model.gguf"), tmp_path)


def test_llama_backend_capabilities(tmp_path: Path) -> None:
    backend = LlamaCppBackend(config(tmp_path), engine_factory=lambda _: FakeGgufEngine())
    assert backend.supported_formats == frozenset({ModelFormat.GGUF})
    assert BackendCapability.CANCELLATION in backend.capabilities


def test_llama_test_engine_load_generate_unload(tmp_path: Path) -> None:
    backend = LlamaCppBackend(config(tmp_path), engine_factory=lambda _: FakeGgufEngine())
    backend.load_model()
    response = backend.generate(request())
    assert response.text == "fake:hello gguf"
    assert response.usage.total_tokens == 3
    backend.unload_model()
    assert not backend.health().details["loaded"]


def test_llama_stream_order(tmp_path: Path) -> None:
    backend = LlamaCppBackend(config(tmp_path), engine_factory=lambda _: FakeGgufEngine())
    backend.load_model()
    events = list(backend.stream_generate(request()))
    assert events[0].type is StreamEventType.STARTED
    assert events[-1].type is StreamEventType.COMPLETED
    assert (
        "".join(item.text for item in events if item.type is StreamEventType.TOKEN) == "fake stream"
    )


def test_llama_stream_cancellation(tmp_path: Path) -> None:
    backend = LlamaCppBackend(config(tmp_path), engine_factory=lambda _: FakeGgufEngine())
    backend.load_model()
    events = backend.stream_generate(request())
    assert next(events).type is StreamEventType.STARTED
    assert backend.cancel("gguf-request")
    assert next(events).type is StreamEventType.CANCELLED


def test_llama_missing_file_rejected(tmp_path: Path) -> None:
    backend = LlamaCppBackend(
        config(tmp_path, create=False), engine_factory=lambda _: FakeGgufEngine()
    )
    with pytest.raises(BackendError):
        backend.load_model()


def test_real_llama_integration_stays_controlled(tmp_path: Path) -> None:
    backend = LlamaCppBackend(config(tmp_path))
    if backend.availability().available:
        with pytest.raises(BackendUnavailableError, match="foundation"):
            backend.load_model()
    else:
        with pytest.raises(BackendUnavailableError, match="Optional dependency"):
            backend.load_model()


def test_llama_tokenizer_is_not_claimed(tmp_path: Path) -> None:
    backend = LlamaCppBackend(config(tmp_path), engine_factory=lambda _: FakeGgufEngine())
    with pytest.raises(BackendError, match="henüz uygulanmadı"):
        backend.create_tokenizer()


def test_llama_resource_report(tmp_path: Path) -> None:
    backend = LlamaCppBackend(config(tmp_path), engine_factory=lambda _: FakeGgufEngine())
    backend.load_model()
    backend.generate(request())
    report = backend.resource_usage()
    assert report.model_load_seconds is not None
    assert report.generated_tokens == 1


def test_llama_metadata_marks_foundation(tmp_path: Path) -> None:
    backend = LlamaCppBackend(config(tmp_path))
    assert backend.metadata["format"] is ModelFormat.GGUF
    assert backend.metadata["foundation_only"] is True
