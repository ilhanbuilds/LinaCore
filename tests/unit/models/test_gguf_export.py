from pathlib import Path

import pytest

from lina_core.core.exceptions import ModelLoadError, SecurityError
from lina_core.models.gguf_export import (
    GgufExportPlan,
    export_gguf,
    gguf_export_availability,
)


def test_gguf_export_requires_explicit_converter(tmp_path: Path) -> None:
    availability = gguf_export_availability(tmp_path / "convert_hf_to_gguf.py")
    assert not availability.available
    plan = GgufExportPlan(Path("model"), Path("model.gguf"), tmp_path / "missing.py")
    with pytest.raises(ModelLoadError, match="açıkça"):
        export_gguf(plan, tmp_path)


def test_gguf_export_rejects_unsafe_format(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        GgufExportPlan(Path("model"), Path("model.gguf"), tmp_path / "tool.py", "q4_k_m")


def test_gguf_export_rejects_escape(tmp_path: Path) -> None:
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("", encoding="utf-8")
    plan = GgufExportPlan(Path("../model"), Path("model.gguf"), converter)
    with pytest.raises(SecurityError):
        export_gguf(plan, tmp_path)
