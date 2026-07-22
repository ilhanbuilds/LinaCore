import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, cast

from lina_core.core.exceptions import ConfigurationError
from lina_core.security.limits import ResourceLimits
from lina_core.security.secrets import redact_mapping

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DirectoryConfig:
    models: str = "models"
    checkpoints: str = "checkpoints"
    datasets: str = "datasets"
    reports: str = "reports"
    logs: str = "logs"


@dataclass(frozen=True, slots=True)
class AppConfig:
    log_level: str = "INFO"
    runtime_provider: str = "mock"
    tokenizer: str = "mock"
    debug: bool = False
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)

    def __post_init__(self) -> None:
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError("Invalid log level")
        if not self.runtime_provider or not self.tokenizer:
            raise ConfigurationError("Provider identifiers cannot be empty")

    def __repr__(self) -> str:
        return (
            f"AppConfig({redact_mapping({f.name: getattr(self, f.name) for f in fields(self)})!r})"
        )


def _checked(cls: type[T], data: dict[str, Any]) -> T:
    expected = {item.name for item in fields(cast(Any, cls))}
    unknown = set(data) - expected
    if unknown:
        raise ConfigurationError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    try:
        return cls(**data)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


def load_config(path: Path | None = None, *, environ: dict[str, str] | None = None) -> AppConfig:
    data: dict[str, Any] = {}
    if path is not None:
        try:
            if path.stat().st_size > ResourceLimits().max_config_size:
                raise ConfigurationError("Configuration file is too large")
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"Cannot load configuration: {exc}") from exc
    raw_limits = data.pop("limits", {})
    raw_dirs = data.pop("directories", {})
    env = environ if environ is not None else os.environ
    overrides = {
        "log_level": env.get("LINA_CORE_LOG_LEVEL"),
        "runtime_provider": env.get("LINA_CORE_PROVIDER"),
        "tokenizer": env.get("LINA_CORE_TOKENIZER"),
    }
    data.update({key: value for key, value in overrides.items() if value is not None})
    return _checked(
        AppConfig,
        {
            **data,
            "limits": _checked(ResourceLimits, raw_limits),
            "directories": _checked(DirectoryConfig, raw_dirs),
        },
    )
