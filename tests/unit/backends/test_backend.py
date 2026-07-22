from collections.abc import Iterator

import pytest

from lina_core.backends.capabilities import BackendCapability, ModelFormat
from lina_core.backends.device import (
    DeviceInfo,
    DeviceType,
    DTypePolicy,
    QuantizationPolicy,
    select_device,
)
from lina_core.backends.errors import (
    BackendError,
    DeviceSelectionError,
    DTypeCompatibilityError,
    GenerationError,
)
from lina_core.backends.transformers.availability import transformers_availability
from lina_core.backends.transformers.config import TransformersBackendConfig
from lina_core.backends.transformers.generation import generation_kwargs
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.inference.sampling import GenerationOptions
from lina_core.inference.stop import apply_stop_sequences
from lina_core.models.mock import MockModelProvider

CPU = DeviceInfo(DeviceType.CPU, display_name="Test CPU")
CUDA = DeviceInfo(DeviceType.CUDA, 0, "Test CUDA", recommended=True)


@pytest.mark.parametrize("capability", list(BackendCapability))
def test_backend_capability_values(capability: BackendCapability) -> None:
    assert BackendCapability(capability.value) is capability


@pytest.mark.parametrize("model_format", list(ModelFormat))
def test_model_format_values(model_format: ModelFormat) -> None:
    assert ModelFormat(model_format.value) is model_format


@pytest.mark.parametrize(
    ("policy", "devices", "expected"),
    [
        ("auto", (CPU,), "cpu"),
        ("auto", (CPU, CUDA), "cuda:0"),
        ("cpu", (CPU, CUDA), "cpu"),
        ("cuda", (CPU, CUDA), "cuda:0"),
        ("cuda:0", (CPU, CUDA), "cuda:0"),
    ],
)
def test_device_selection(policy: str, devices: tuple[DeviceInfo, ...], expected: str) -> None:
    assert select_device(policy, devices).device.identifier == expected


@pytest.mark.parametrize("policy", ["cuda", "cuda:1", "mps", "directml", "cuda:nope"])
def test_invalid_device_selection(policy: str) -> None:
    with pytest.raises(DeviceSelectionError):
        select_device(policy, (CPU,))


def test_empty_device_detection_rejected() -> None:
    with pytest.raises(DeviceSelectionError):
        select_device("auto", ())


@pytest.mark.parametrize(
    ("device", "dtype", "expected"),
    [
        (CPU, DTypePolicy.AUTO, DTypePolicy.FLOAT32),
        (CPU, DTypePolicy.FLOAT32, DTypePolicy.FLOAT32),
        (CUDA, DTypePolicy.AUTO, DTypePolicy.FLOAT16),
        (CUDA, DTypePolicy.BFLOAT16, DTypePolicy.BFLOAT16),
    ],
)
def test_dtype_selection(device: DeviceInfo, dtype: DTypePolicy, expected: DTypePolicy) -> None:
    assert select_device(device.identifier, (CPU, CUDA), dtype).dtype is expected


def test_cpu_float16_rejected() -> None:
    with pytest.raises(DTypeCompatibilityError):
        select_device("cpu", (CPU,), DTypePolicy.FLOAT16)


@pytest.mark.parametrize("quantization", [QuantizationPolicy.INT8, QuantizationPolicy.INT4])
def test_optional_quantization_rejected(quantization: QuantizationPolicy) -> None:
    with pytest.raises(DTypeCompatibilityError):
        select_device("cpu", (CPU,), quantization=quantization)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_path": ""},
        {"model_path": "x", "trust_remote_code": True},
        {"model_path": "x", "local_files_only": False},
        {"model_path": "x", "revision": ""},
        {"model_path": "x", "chat_template_policy": "silent"},
    ],
)
def test_transformers_config_rejects_unsafe_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(BackendError):
        TransformersBackendConfig(**kwargs)  # type: ignore[arg-type]


def test_remote_config_requires_explicit_permission() -> None:
    config = TransformersBackendConfig(
        "organization/model",
        local_files_only=False,
        allow_remote_downloads=True,
        revision="abc123",
    )
    assert config.allow_remote_downloads and not config.trust_remote_code


@pytest.mark.parametrize(
    ("temperature", "sampling"),
    [(0.0, False), (0.1, True), (0.7, True), (1.0, True), (2.0, True)],
)
def test_generation_sampling_mapping(temperature: float, sampling: bool) -> None:
    values = generation_kwargs(GenerationOptions(temperature=temperature))
    assert values["do_sample"] is sampling


def test_greedy_rejects_custom_sampling_options() -> None:
    with pytest.raises(GenerationError):
        generation_kwargs(GenerationOptions(temperature=0, top_k=1))


@pytest.mark.parametrize(
    ("text", "stops", "expected", "stopped"),
    [
        ("hello", (), "hello", False),
        ("hello END ignored", ("END",), "hello ", True),
        ("a X b Y", ("Y", "X"), "a ", True),
        ("same", ("",), "same", False),
        ("Türkçe dur sonra", ("dur",), "Türkçe ", True),
    ],
)
def test_stop_sequences(text: str, stops: tuple[str, ...], expected: str, stopped: bool) -> None:
    result = apply_stop_sequences(text, stops)
    assert (result.text, result.stopped) == (expected, stopped)


def test_transformers_availability_is_typed() -> None:
    result = transformers_availability()
    assert result.backend_id == "transformers"
    assert isinstance(result.available, bool)
    assert isinstance(result.missing_packages, tuple)


def test_transformers_unavailable_message_is_controlled() -> None:
    result = transformers_availability()
    if not result.available:
        assert "optional dependency" in result.reason
        assert "ImportError" not in result.reason


def test_mock_generation_still_works() -> None:
    provider = MockModelProvider()
    provider.load()
    response = provider.generate(InferenceRequest((ChatMessage(ChatRole.USER, "baseline"),)))
    assert response.text == "[mock] baseline"


def test_mock_stream_terminal_event() -> None:
    provider = MockModelProvider()
    provider.load()
    events: Iterator[object] = provider.stream_generate(
        InferenceRequest((ChatMessage(ChatRole.USER, "stream"),))
    )
    assert list(events)[-1].type == "completed"  # type: ignore[attr-defined]
