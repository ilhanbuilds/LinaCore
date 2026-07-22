import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from lina_core.core.exceptions import ModelLoadError
from lina_core.security.paths import safe_resolve


@dataclass(frozen=True, slots=True)
class GgufExportPlan:
    source_model: Path
    output_file: Path
    converter: Path
    output_type: str = "f16"
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.output_type not in {"f16", "f32"}:
            raise ModelLoadError("İlk export aşamasında yalnız f16 ve f32 destekleniyor.")
        if self.output_file.suffix.casefold() != ".gguf":
            raise ModelLoadError("GGUF output dosyası .gguf uzantılı olmalıdır.")


@dataclass(frozen=True, slots=True)
class GgufExportAvailability:
    available: bool
    reason: str
    converter: str | None


def gguf_export_availability(converter: Path) -> GgufExportAvailability:
    available = converter.is_file() and converter.name == "convert_hf_to_gguf.py"
    return GgufExportAvailability(
        available,
        "llama.cpp conversion aracı hazır."
        if available
        else "convert_hf_to_gguf.py yolu açıkça sağlanmalı; araç otomatik indirilmez.",
        converter.name if available else None,
    )


def export_gguf(plan: GgufExportPlan, model_root: Path) -> Path:
    source = safe_resolve(model_root, plan.source_model)
    output = safe_resolve(model_root, plan.output_file)
    availability = gguf_export_availability(plan.converter)
    if not availability.available:
        raise ModelLoadError(availability.reason)
    if not source.is_dir():
        raise ModelLoadError("Export kaynak model dizini bulunamadı.")
    if output.exists() and not plan.overwrite:
        raise ModelLoadError("GGUF output mevcut; açık overwrite izni gerekli.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(plan.converter.resolve()),
            str(source),
            "--outfile",
            str(output),
            "--outtype",
            plan.output_type,
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode != 0:
        raise ModelLoadError(
            "GGUF conversion aracı başarısız oldu; debug logunu yerelde inceleyin."
        )
    if not output.is_file():
        raise ModelLoadError("Conversion başarılı görünse de GGUF output oluşmadı.")
    return output
