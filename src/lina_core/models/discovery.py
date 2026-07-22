from dataclasses import dataclass
from pathlib import Path

from lina_core.backends.llama_cpp.availability import llama_cpp_availability
from lina_core.backends.transformers.availability import transformers_availability
from lina_core.models.manifest import MANIFEST_NAME
from lina_core.models.validation import ModelValidation, load_model_manifest


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    model_id: str
    display_name: str
    backend: str
    model_format: str
    status: str
    device: str
    license: str
    manifest_path: Path
    validation: ModelValidation
    backend_available: bool


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    models: tuple[DiscoveredModel, ...]
    invalid_manifests: tuple[Path, ...]
    duplicate_ids: tuple[str, ...]


def discover_models(model_root: Path) -> DiscoveryResult:
    root = model_root.resolve()
    if not root.exists():
        return DiscoveryResult((), (), ())
    found: list[DiscoveredModel] = []
    invalid: list[Path] = []
    duplicates: set[str] = set()
    seen: set[str] = set()
    backend_statuses = {
        "transformers": transformers_availability(),
        "llama.cpp": llama_cpp_availability(),
    }
    for path in sorted(root.rglob(MANIFEST_NAME)):
        validation = load_model_manifest(path, root)
        if validation.manifest is None:
            invalid.append(path)
            continue
        manifest = validation.manifest
        backend_status = backend_statuses[manifest.backend]
        if manifest.model_id in seen:
            duplicates.add(manifest.model_id)
            continue
        seen.add(manifest.model_id)
        status = validation.status
        if status == "available" and not backend_status.available:
            status = "backend-unavailable"
        found.append(
            DiscoveredModel(
                manifest.model_id,
                manifest.display_name,
                manifest.backend,
                manifest.model_format,
                status,
                manifest.default_device,
                manifest.license,
                path,
                validation,
                backend_status.available,
            )
        )
    return DiscoveryResult(tuple(found), tuple(invalid), tuple(sorted(duplicates)))
