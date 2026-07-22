import tomllib
from dataclasses import dataclass
from pathlib import Path

from lina_core.core.exceptions import ConfigurationError, LinaCoreError
from lina_core.models.manifest import ModelManifest, coerce_manifest
from lina_core.security.paths import safe_resolve


@dataclass(frozen=True, slots=True)
class ModelValidation:
    valid: bool
    status: str
    manifest: ModelManifest | None = None
    model_path: Path | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None


def load_model_manifest(
    manifest_path: Path,
    model_root: Path,
    *,
    max_size: int = 262_144,
) -> ModelValidation:
    root = model_root.resolve()
    try:
        resolved_manifest = manifest_path.resolve()
        if resolved_manifest != root and root not in resolved_manifest.parents:
            raise ConfigurationError("Manifest model root dışına çıkıyor.")
        if resolved_manifest.stat().st_size > max_size:
            raise ConfigurationError("Model manifesti izin verilen boyutu aşıyor.")
        raw = tomllib.loads(resolved_manifest.read_text(encoding="utf-8"))
        manifest = coerce_manifest(raw.get("model", raw))
        model_path = safe_resolve(resolved_manifest.parent, manifest.model_path)
        if model_path != root and root not in model_path.parents:
            raise ConfigurationError("Model yolu yapılandırılmış model root dışına çıkıyor.")
        if manifest.tokenizer_path:
            tokenizer_path = safe_resolve(resolved_manifest.parent, manifest.tokenizer_path)
            if tokenizer_path != root and root not in tokenizer_path.parents:
                raise ConfigurationError("Tokenizer yolu model root dışına çıkıyor.")
        if manifest.backend == "llama.cpp" and model_path.suffix.casefold() != ".gguf":
            raise ConfigurationError("llama.cpp manifesti .gguf model dosyasına işaret etmelidir.")
        if manifest.example:
            return ModelValidation(
                True, "example", manifest, model_path, ("Bu bir örnek manifesttir.",)
            )
        if not model_path.exists():
            return ModelValidation(
                False, "missing-files", manifest, model_path, error="Model dosyaları bulunamadı."
            )
        return ModelValidation(True, "available", manifest, model_path)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, LinaCoreError) as exc:
        return ModelValidation(False, "invalid-manifest", error=str(exc))
