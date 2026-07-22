from collections.abc import Iterator
from pathlib import Path

import pytest

from lina_core.backends.transformers.config import TransformersBackendConfig
from lina_core.backends.transformers.model import TransformersModelProvider
from lina_core.core.exceptions import SecurityError
from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse, TokenUsage
from lina_core.inference.stop import apply_stop_sequences
from lina_core.inference.streaming import StreamEvent, StreamEventType
from lina_core.models.compatibility import (
    CompatibilityStatus,
    ModelCompatibilityReport,
    TransformersCompatibilityRunner,
    write_compatibility_report,
)
from lina_core.runtime.health import HealthLevel, HealthStatus
from lina_core.runtime.resources import ResourceReport


class CompatibilityProviderDouble(TransformersModelProvider):
    def __init__(self, *, fail_generation: bool = False) -> None:
        super().__init__(
            TransformersBackendConfig(
                "C:/private/models/test-model",
                model_id="compatibility-double",
                chat_template_policy="fallback",
            )
        )
        self.loaded = False
        self.cancelled = False
        self.fail_generation = fail_generation
        self.generated_tokens = 0

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        if self.fail_generation:
            raise RuntimeError(f"failed at {self.config.model_path}")
        base = " alpha beta"
        stopped = apply_stop_sequences(base, request.options.stop)
        tokens = len(stopped.text.split())
        self.generated_tokens += tokens
        return InferenceResponse(
            request.request_id,
            self.config.model_id,
            stopped.text,
            "stop" if stopped.stopped else "length",
            TokenUsage(3, tokens),
            0.01,
        )

    def stream_generate(self, request: InferenceRequest) -> Iterator[StreamEvent]:
        self.cancelled = False
        yield StreamEvent(StreamEventType.STARTED, request.request_id, index=0)
        if self.cancelled:
            yield StreamEvent(StreamEventType.CANCELLED, request.request_id, index=1)
            return
        self.generated_tokens += 2
        yield StreamEvent(StreamEventType.TOKEN, request.request_id, "alpha ", 1)
        if self.cancelled:
            yield StreamEvent(StreamEventType.CANCELLED, request.request_id, index=2)
            return
        yield StreamEvent(StreamEventType.TOKEN, request.request_id, "beta", 2)
        yield StreamEvent(StreamEventType.COMPLETED, request.request_id, index=3)

    def cancel(self, request_id: str, reason: str = "user") -> bool:
        del request_id, reason
        self.cancelled = True
        return True

    def health(self) -> HealthStatus:
        return HealthStatus(
            HealthLevel.HEALTHY if self.loaded else HealthLevel.DEGRADED,
            "test",
            {
                "loaded": self.loaded,
                "device": "cpu",
                "dtype": "float32",
            },
        )

    def resource_usage(self) -> ResourceReport:
        return ResourceReport(
            model_load_seconds=0.01,
            generated_tokens=self.generated_tokens,
            last_inference_latency=0.01,
        )


def runner(provider: CompatibilityProviderDouble) -> TransformersCompatibilityRunner:
    return TransformersCompatibilityRunner(
        provider,
        model_version="test",
        source_revision="abc123",
        license_name="Apache-2.0",
    )


def test_compatibility_runner_passes_all_stages() -> None:
    report = runner(CompatibilityProviderDouble()).run()
    assert report.status is CompatibilityStatus.PASSED
    assert [check.name for check in report.checks] == [
        "load",
        "generation",
        "streaming",
        "cancellation",
        "stop_sequence",
        "health",
        "resources",
        "unload",
    ]
    assert all(check.status is CompatibilityStatus.PASSED for check in report.checks)


def test_compatibility_report_contains_required_identity() -> None:
    report = runner(CompatibilityProviderDouble()).run()
    assert report.model_id == "compatibility-double"
    assert report.model_version == "test"
    assert report.source_revision == "abc123"
    assert report.license == "Apache-2.0"
    assert report.device == "cpu"
    assert report.dtype == "float32"


def test_compatibility_failure_is_typed_and_path_redacted() -> None:
    report = runner(CompatibilityProviderDouble(fail_generation=True)).run()
    assert report.status is CompatibilityStatus.FAILED
    assert report.error is not None
    assert "C:/private" not in report.error
    assert report.checks[-1].name == "unload"


def test_compatibility_report_json_serializable() -> None:
    payload = runner(CompatibilityProviderDouble()).run().to_json()
    assert '"status": "passed"' in payload
    assert '"license": "Apache-2.0"' in payload


def test_write_compatibility_report(tmp_path: Path) -> None:
    report = runner(CompatibilityProviderDouble()).run()
    target = write_compatibility_report(report, tmp_path, "compatibility.json")
    assert target.exists() and "compatibility-double" in target.read_text(encoding="utf-8")


def test_report_path_escape_rejected(tmp_path: Path) -> None:
    report = runner(CompatibilityProviderDouble()).run()
    with pytest.raises(SecurityError):
        write_compatibility_report(report, tmp_path, "../outside.json")


def test_report_dataclass_is_typed() -> None:
    report = runner(CompatibilityProviderDouble()).run()
    assert isinstance(report, ModelCompatibilityReport)
    assert report.schema_version == 1
