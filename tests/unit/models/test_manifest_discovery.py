from pathlib import Path

import pytest

from lina_core.core.exceptions import ConfigurationError
from lina_core.models.discovery import discover_models
from lina_core.models.manifest import ModelManifest, coerce_manifest
from lina_core.models.validation import load_model_manifest

BASE: dict[str, object] = {
    "schema_version": 1,
    "model_id": "test-model",
    "display_name": "Test Model",
    "version": "1",
    "backend": "transformers",
    "architecture": "causal-lm",
    "model_path": "files",
    "source": "local-test",
    "license": "test-only",
    "context_length": 2048,
}


def manifest_text(model_id: str = "test-model", **overrides: object) -> str:
    values = {**BASE, "model_id": model_id, **overrides}
    lines = ["[model]"]
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, int):
            rendered = str(value)
        elif isinstance(value, list):
            rendered = "[" + ", ".join(f'"{item}"' for item in value) + "]"
        else:
            rendered = f'"{value}"'
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines)


def write_manifest(root: Path, folder: str = "one", **overrides: object) -> Path:
    directory = root / folder
    directory.mkdir(parents=True)
    path = directory / "linacore-model.toml"
    path.write_text(manifest_text(**overrides), encoding="utf-8")
    return path


def test_valid_manifest_dataclass() -> None:
    manifest = coerce_manifest(dict(BASE))
    assert isinstance(manifest, ModelManifest)
    assert manifest.model_id == "test-model"


@pytest.mark.parametrize(
    "mutation",
    [
        {"unknown": "x"},
        {"schema_version": 2},
        {"backend": "gguf"},
        {"model_id": ""},
        {"display_name": ""},
        {"model_path": ""},
        {"context_length": 0},
        {"trust_remote_code": True},
        {"local_files_only": False},
        {"source_revision": "bad revision!"},
        {"default_dtype": "float64"},
        {"quantization": "int3"},
        {"capabilities": ["vision"]},
        {"chat_template_policy": "silent"},
        {"checksums": {"model": "md5:bad"}},
    ],
)
def test_invalid_manifest_values(mutation: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        coerce_manifest({**BASE, **mutation})


def test_manifest_missing_files(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)
    result = load_model_manifest(path, tmp_path)
    assert not result.valid and result.status == "missing-files"


def test_manifest_available(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)
    (path.parent / "files").mkdir()
    result = load_model_manifest(path, tmp_path)
    assert result.valid and result.status == "available"


def test_manifest_example_is_not_available_model(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, example=True)
    result = load_model_manifest(path, tmp_path)
    assert result.valid and result.status == "example" and result.warnings


def test_manifest_outside_root_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.toml"
    outside.write_text(manifest_text(), encoding="utf-8")
    assert load_model_manifest(outside, root).status == "invalid-manifest"


def test_manifest_path_traversal_rejected(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, model_path="../../escape")
    assert load_model_manifest(path, tmp_path).status == "invalid-manifest"


def test_manifest_size_bounded(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)
    assert load_model_manifest(path, tmp_path, max_size=10).status == "invalid-manifest"


def test_empty_discovery(tmp_path: Path) -> None:
    result = discover_models(tmp_path)
    assert result.models == ()


def test_missing_root_discovery(tmp_path: Path) -> None:
    result = discover_models(tmp_path / "missing")
    assert not result.models and not result.invalid_manifests


def test_discovery_finds_manifest_lazily(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, example=True)
    result = discover_models(tmp_path)
    assert result.models[0].manifest_path == path
    assert result.models[0].status == "example"


def test_discovery_reports_invalid_manifest(tmp_path: Path) -> None:
    path = tmp_path / "bad" / "linacore-model.toml"
    path.parent.mkdir()
    path.write_text("bad=[", encoding="utf-8")
    result = discover_models(tmp_path)
    assert result.invalid_manifests == (path,)


def test_discovery_duplicate_ids(tmp_path: Path) -> None:
    write_manifest(tmp_path, "one", model_id="duplicate", example=True)
    write_manifest(tmp_path, "two", model_id="duplicate", example=True)
    result = discover_models(tmp_path)
    assert result.duplicate_ids == ("duplicate",)
    assert len(result.models) == 1


def test_discovery_does_not_read_weights(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, example=True)
    weight = path.parent / "model.safetensors"
    weight.write_bytes(b"not-a-real-weight")
    result = discover_models(tmp_path)
    assert result.models and weight.read_bytes() == b"not-a-real-weight"


def test_example_repository_manifest() -> None:
    root = Path("examples/models").resolve()
    result = discover_models(root)
    assert result.models[0].status == "example"
