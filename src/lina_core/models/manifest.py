import re
from dataclasses import dataclass, field, fields
from typing import Any

from lina_core.backends.capabilities import ModelFormat
from lina_core.backends.device import DTypePolicy, QuantizationPolicy
from lina_core.core.exceptions import ConfigurationError
from lina_core.models.metadata import ModelCapability

MANIFEST_NAME = "linacore-model.toml"
SUPPORTED_BACKENDS = {"transformers"}
_REVISION = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")
_CHECKSUM = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    schema_version: int
    model_id: str
    display_name: str
    version: str
    backend: str
    architecture: str
    model_path: str
    source: str
    license: str
    context_length: int
    default_device: str = "auto"
    default_dtype: DTypePolicy = DTypePolicy.AUTO
    quantization: QuantizationPolicy = QuantizationPolicy.NONE
    capabilities: tuple[ModelCapability, ...] = (ModelCapability.TEXT_GENERATION,)
    chat_template_policy: str = "required"
    trust_remote_code: bool = False
    local_files_only: bool = True
    tokenizer_path: str | None = None
    source_revision: str | None = None
    parameter_count: int | None = None
    checksums: dict[str, str] = field(default_factory=dict)
    notes: str | None = None
    example: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConfigurationError("Desteklenmeyen model manifest schema sürümü.")
        if self.backend not in SUPPORTED_BACKENDS:
            raise ConfigurationError(f"Desteklenmeyen backend: {self.backend}")
        if not self.model_id or not self.display_name or not self.model_path:
            raise ConfigurationError("Model kimliği, adı ve yolu zorunludur.")
        if self.context_length <= 0:
            raise ConfigurationError("Context length pozitif olmalıdır.")
        if self.trust_remote_code:
            raise ConfigurationError("trust_remote_code güvenlik politikası gereği reddedildi.")
        if not self.local_files_only:
            raise ConfigurationError("Manifest varsayılan olarak local_files_only olmalıdır.")
        if self.source_revision is not None and not _REVISION.fullmatch(self.source_revision):
            raise ConfigurationError("Geçersiz source revision.")
        if any(not _CHECKSUM.fullmatch(value) for value in self.checksums.values()):
            raise ConfigurationError("Checksum değerleri sha256 biçiminde olmalıdır.")
        unsupported = set(self.capabilities) - {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.STREAMING,
        }
        if unsupported:
            raise ConfigurationError("Manifest desteklenmeyen capability içeriyor.")
        if self.chat_template_policy not in {"required", "fallback"}:
            raise ConfigurationError("Geçersiz chat template politikası.")

    @property
    def model_format(self) -> ModelFormat:
        return ModelFormat.TRANSFORMERS


def manifest_fields() -> frozenset[str]:
    return frozenset(item.name for item in fields(ModelManifest))


def coerce_manifest(data: dict[str, Any]) -> ModelManifest:
    unknown = set(data) - manifest_fields()
    if unknown:
        raise ConfigurationError(f"Bilinmeyen manifest alanları: {', '.join(sorted(unknown))}")
    try:
        converted = dict(data)
        converted["default_dtype"] = DTypePolicy(converted.get("default_dtype", "auto"))
        converted["quantization"] = QuantizationPolicy(converted.get("quantization", "none"))
        converted["capabilities"] = tuple(
            ModelCapability(value) for value in converted.get("capabilities", ["text_generation"])
        )
        return ModelManifest(**converted)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Geçersiz model manifesti: {exc}") from exc
