from pathlib import Path

from lina_core.backends.transformers.config import TransformersBackendConfig
from lina_core.backends.transformers.model import TransformersModelProvider
from lina_core.core.exceptions import ModelLoadError
from lina_core.models.validation import load_model_manifest
from lina_core.security.paths import safe_resolve


def provider_from_manifest(path: Path, model_root: Path) -> TransformersModelProvider:
    validation = load_model_manifest(path, model_root)
    if not validation.valid or validation.manifest is None or validation.model_path is None:
        raise ModelLoadError(validation.error or f"Model kullanılamıyor: {validation.status}")
    manifest = validation.manifest
    if manifest.example:
        raise ModelLoadError("Örnek manifest gerçek model olarak yüklenemez.")
    if manifest.backend != "transformers":
        raise ModelLoadError(
            "Bu helper yalnız Transformers provider oluşturur; GGUF foundation henüz yüklenemez."
        )
    tokenizer_path = (
        str(safe_resolve(path.parent.resolve(), manifest.tokenizer_path))
        if manifest.tokenizer_path
        else None
    )
    return TransformersModelProvider(
        TransformersBackendConfig(
            model_path=str(validation.model_path),
            tokenizer_path=tokenizer_path,
            model_id=manifest.model_id,
            display_name=manifest.display_name,
            revision=manifest.source_revision,
            device=manifest.default_device,
            dtype=manifest.default_dtype,
            quantization=manifest.quantization,
            local_files_only=manifest.local_files_only,
            trust_remote_code=manifest.trust_remote_code,
            chat_template_policy=manifest.chat_template_policy,
        ),
        context_length=manifest.context_length,
        architecture=manifest.architecture,
        source=manifest.source,
        license_name=manifest.license,
        capabilities=frozenset(manifest.capabilities),
    )
