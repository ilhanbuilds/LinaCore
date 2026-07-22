from pathlib import Path

import pytest

from lina_core.cli import main
from lina_core.core.config import load_config
from lina_core.core.exceptions import ConfigurationError, EvaluationError
from lina_core.diagnostics.environment import environment_report
from lina_core.diagnostics.report import diagnostics_json, diagnostics_text
from lina_core.evaluation.benchmark import run_benchmark
from lina_core.models.mock import MockModelProvider
from lina_core.runtime.cancellation import CancellationRegistry, CancellationToken
from lina_core.runtime.resources import ResourceReport, process_memory_bytes
from lina_core.runtime.service import RuntimeService


def test_cancellation_token_initial_state() -> None:
    token = CancellationToken("r")
    assert not token.cancelled and not token.state.cancelled


def test_cancellation_token_cancel() -> None:
    token = CancellationToken("r")
    assert token.cancel("user") and token.state.reason == "user"


def test_duplicate_cancellation_is_idempotent() -> None:
    token = CancellationToken("r")
    assert token.cancel() and not token.cancel()


def test_cancellation_records_timestamp() -> None:
    token = CancellationToken("r")
    token.cancel()
    assert token.state.requested_at is not None


def test_registry_create_is_idempotent() -> None:
    registry = CancellationRegistry()
    assert registry.create("r") is registry.create("r")


def test_registry_unknown_cancel() -> None:
    assert not CancellationRegistry().cancel("missing")


def test_registry_cancel_known() -> None:
    registry = CancellationRegistry()
    registry.create("r")
    assert registry.cancel("r")


def test_registry_finish() -> None:
    registry = CancellationRegistry()
    registry.create("r")
    registry.finish("r")
    assert registry.active_count == 0


@pytest.mark.parametrize("count", [1, 2, 5, 10])
def test_registry_cancel_all(count: int) -> None:
    registry = CancellationRegistry()
    for index in range(count):
        registry.create(str(index))
    assert registry.cancel_all("shutdown") == count


def test_runtime_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError):
        RuntimeService(max_concurrent_requests=0)


def test_runtime_health_is_expanded() -> None:
    health = RuntimeService().health()
    assert "runtime_state" in health.details
    assert "current_requests" in health.details
    assert "uptime_seconds" in health.details


def test_runtime_resource_report_before_start() -> None:
    report = RuntimeService().resource_usage()
    assert isinstance(report, ResourceReport)
    assert report.active_requests == 0


def test_process_memory_fallback_is_safe() -> None:
    value = process_memory_bytes()
    assert value is None or value > 0


def test_resource_report_serializable() -> None:
    assert ResourceReport(generated_tokens=2).to_dict()["generated_tokens"] == 2


def test_mock_benchmark_rejected() -> None:
    provider = MockModelProvider()
    provider.load()
    with pytest.raises(EvaluationError):
        run_benchmark(provider)


def test_environment_report_redacts_paths(tmp_path: Path) -> None:
    report = environment_report(tmp_path / "private" / "models")
    assert report["model_root"] == "models"
    assert str(tmp_path) not in str(report)


def test_environment_report_has_security_defaults(tmp_path: Path) -> None:
    report = environment_report(tmp_path)
    assert report["security"]["trust_remote_code"] is False


def test_diagnostics_json_serializable(tmp_path: Path) -> None:
    assert "linacore_version" in diagnostics_json(environment_report(tmp_path))


def test_diagnostics_text_is_turkish(tmp_path: Path) -> None:
    assert "İşletim sistemi" in diagnostics_text(environment_report(tmp_path))


@pytest.mark.parametrize(
    "args",
    [
        ["diagnostics"],
        ["diagnostics", "--json"],
        ["backends", "list"],
        ["devices", "list"],
        ["models", "discover"],
        ["models", "list"],
        ["runtime", "unload"],
    ],
)
def test_new_cli_commands(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(args) == 0
    assert capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["models", "validate", "missing"],
        ["runtime", "load", "missing"],
        ["inference", "generate", "--prompt", "hello", "--model", "missing"],
        ["inference", "stream", "--prompt", "hello", "--model", "missing"],
        ["benchmark", "run"],
    ],
)
def test_new_cli_controlled_errors(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(args) == 2
    assert "error" in capsys.readouterr().err


def test_nested_config_defaults() -> None:
    config = load_config()
    assert config.runtime.model_root == "models"
    assert config.backend.transformers.local_files_only
    assert not config.backend.transformers.trust_remote_code


@pytest.mark.parametrize(
    "content",
    [
        "[runtime]\nmax_concurrent_requests=0",
        "[runtime]\nrequest_timeout_seconds=0",
        "[backend.transformers]\ntrust_remote_code=true",
        "[backend.transformers]\nlocal_files_only=false",
        "[backend.transformers]\ndtype='float64'",
        "[backend.transformers]\nquantization='int3'",
        "[generation]\nmax_new_tokens=0",
        "[generation]\ntruncation_policy='silent'",
        "[security]\nmax_model_manifest_size=0",
        "[security]\nremote_model_policy='allow'",
    ],
)
def test_nested_config_invalid(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


@pytest.mark.model_integration
def test_manual_local_model_integration() -> None:
    import os

    from lina_core.models.compatibility import (
        CompatibilityStatus,
        TransformersCompatibilityRunner,
    )
    from lina_core.models.local import provider_from_manifest
    from lina_core.models.validation import load_model_manifest

    raw_manifest = os.environ.get("LINA_CORE_MODEL_MANIFEST")
    if not raw_manifest:
        pytest.skip("LINA_CORE_MODEL_MANIFEST kullanıcı tarafından sağlanmadı.")
    manifest_path = Path(raw_manifest).resolve()
    model_root = Path(os.environ.get("LINA_CORE_MODEL_ROOT", "models")).resolve()
    validation = load_model_manifest(manifest_path, model_root)
    assert validation.valid and validation.manifest is not None
    manifest = validation.manifest
    report = TransformersCompatibilityRunner(
        provider_from_manifest(manifest_path, model_root),
        model_version=manifest.version,
        source_revision=manifest.source_revision,
        license_name=manifest.license,
    ).run()
    assert report.status is CompatibilityStatus.PASSED, report.to_json()
