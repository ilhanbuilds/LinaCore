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
class RuntimeConfig:
    default_backend: str = "mock"
    default_model_id: str | None = None
    model_root: str = "models"
    lazy_load: bool = True
    max_concurrent_requests: int = 1
    request_timeout_seconds: float = 120.0
    shutdown_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.max_concurrent_requests <= 0:
            raise ConfigurationError("max_concurrent_requests pozitif olmalıdır.")
        if self.request_timeout_seconds <= 0 or self.shutdown_timeout_seconds <= 0:
            raise ConfigurationError("Runtime timeout değerleri pozitif olmalıdır.")


@dataclass(frozen=True, slots=True)
class TransformersConfig:
    enabled: bool = False
    local_files_only: bool = True
    allow_remote_downloads: bool = False
    trust_remote_code: bool = False
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = "none"
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    streaming: bool = True

    def __post_init__(self) -> None:
        if self.trust_remote_code:
            raise ConfigurationError("trust_remote_code etkinleştirilemez.")
        if not self.local_files_only and not self.allow_remote_downloads:
            raise ConfigurationError("Uzak erişim için açık izin gerekir.")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ConfigurationError("Geçersiz dtype.")
        if self.quantization not in {"none", "int8", "int4"}:
            raise ConfigurationError("Geçersiz quantization.")


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    enabled: bool = False
    context_length: int = 2048
    threads: int | None = None
    gpu_layers: int = 0

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ConfigurationError("llama.cpp context length pozitif olmalıdır.")
        if self.threads is not None and self.threads <= 0:
            raise ConfigurationError("llama.cpp thread sayısı pozitif olmalıdır.")
        if self.gpu_layers < 0:
            raise ConfigurationError("llama.cpp GPU layer sayısı negatif olamaz.")


@dataclass(frozen=True, slots=True)
class BackendConfig:
    transformers: TransformersConfig = field(default_factory=TransformersConfig)
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.0
    max_context_policy: str = "error"
    truncation_policy: str = "error"

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ConfigurationError("Generation token sınırı pozitif olmalıdır.")
        if self.max_context_policy not in {"error", "truncate"}:
            raise ConfigurationError("Geçersiz context politikası.")
        if self.truncation_policy not in {"error", "oldest"}:
            raise ConfigurationError("Geçersiz truncation politikası.")


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    allowed_model_roots: tuple[str, ...] = ("models",)
    allow_symlinks: bool = False
    max_model_manifest_size: int = 262_144
    remote_model_policy: str = "deny"

    def __post_init__(self) -> None:
        if self.max_model_manifest_size <= 0:
            raise ConfigurationError("Manifest boyut sınırı pozitif olmalıdır.")
        if self.remote_model_policy not in {"deny", "explicit"}:
            raise ConfigurationError("Geçersiz remote model politikası.")


@dataclass(frozen=True, slots=True)
class AppConfig:
    log_level: str = "INFO"
    runtime_provider: str = "mock"
    tokenizer: str = "mock"
    debug: bool = False
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

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
    raw_runtime = data.pop("runtime", {})
    raw_backend = data.pop("backend", {})
    raw_transformers = raw_backend.pop("transformers", {})
    raw_llama_cpp = raw_backend.pop("llama_cpp", {})
    if raw_backend:
        raise ConfigurationError(f"Bilinmeyen backend alanları: {', '.join(sorted(raw_backend))}")
    raw_generation = data.pop("generation", {})
    raw_security = data.pop("security", {})
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
            "runtime": _checked(RuntimeConfig, raw_runtime),
            "backend": BackendConfig(
                _checked(TransformersConfig, raw_transformers),
                _checked(LlamaCppConfig, raw_llama_cpp),
            ),
            "generation": _checked(GenerationConfig, raw_generation),
            "security": _checked(
                SecurityConfig,
                {
                    **raw_security,
                    "allowed_model_roots": tuple(
                        raw_security.get("allowed_model_roots", ("models",))
                    ),
                },
            ),
        },
    )
