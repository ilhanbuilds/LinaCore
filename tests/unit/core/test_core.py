from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lina_core.core.config import AppConfig, load_config
from lina_core.core.exceptions import ConfigurationError, LinaCoreError, SecurityError
from lina_core.core.lifecycle import Lifecycle, LifecycleState
from lina_core.core.logging import configure_logging
from lina_core.core.paths import ProjectPaths
from lina_core.security.limits import ResourceLimits
from lina_core.security.paths import safe_resolve
from lina_core.security.secrets import contains_secret, redact


def test_config_defaults():
    assert AppConfig().runtime_provider == "mock"


def test_config_is_frozen():
    with pytest.raises(FrozenInstanceError):
        AppConfig().log_level = "DEBUG"  # type: ignore[misc]


def test_load_toml(tmp_path: Path):
    p = tmp_path / "x.toml"
    p.write_text('log_level="DEBUG"')
    assert load_config(p).log_level == "DEBUG"


def test_environment_override():
    assert load_config(environ={"LINA_CORE_PROVIDER": "other"}).runtime_provider == "other"


@pytest.mark.parametrize("content", ["bad=[", "unknown=1", 'log_level="LOUD"'])
def test_invalid_configs(tmp_path: Path, content: str):
    p = tmp_path / "x.toml"
    p.write_text(content)
    with pytest.raises(ConfigurationError):
        load_config(p)


@pytest.mark.parametrize(
    "candidate", ["../escape", "sub/../../escape", str(Path.cwd().anchor + "escape")]
)
def test_unsafe_paths(tmp_path: Path, candidate: str):
    with pytest.raises(SecurityError):
        safe_resolve(tmp_path, candidate)


@pytest.mark.parametrize("candidate", ["file.txt", "nested/file.txt", "."])
def test_safe_paths(tmp_path: Path, candidate: str):
    assert safe_resolve(tmp_path, candidate).is_relative_to(tmp_path.resolve())


def test_project_paths(tmp_path: Path):
    assert ProjectPaths(tmp_path).model_dir == tmp_path / "models"


def test_lifecycle_happy_path():
    life = Lifecycle()
    life.start()
    assert life.state is LifecycleState.READY
    life.stop()
    assert life.state is LifecycleState.STOPPED


def test_lifecycle_invalid():
    with pytest.raises(LinaCoreError):
        Lifecycle().transition(LifecycleState.READY)


def test_lifecycle_context():
    life = Lifecycle()
    with life:
        assert life.state is LifecycleState.READY
    assert life.state is LifecycleState.STOPPED


def test_logging_deduplicates():
    logger = configure_logging()
    before = len(logger.handlers)
    configure_logging()
    assert len(logger.handlers) == before


def test_exception_hierarchy():
    assert issubclass(ConfigurationError, LinaCoreError)


@pytest.mark.parametrize(
    "value",
    [
        "api_key=abcdefghijklm",
        "Bearer abcdefghijklmnop",
        "password: hunter2",
        "-----BEGIN PRIVATE KEY-----",
    ],
)
def test_secret_detection(value: str):
    assert contains_secret(value)


@pytest.mark.parametrize(
    "value", ["token count is 5", "password policy", "a normal technical example"]
)
def test_secret_false_positives(value: str):
    assert not contains_secret(value)


def test_redaction():
    assert "hunter2" not in redact("password=hunter2")


def test_limits_positive():
    assert ResourceLimits().max_messages > 0


def test_limits_reject_zero():
    with pytest.raises(ValueError):
        ResourceLimits(max_messages=0)
