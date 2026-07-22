import json
from collections.abc import Mapping
from typing import Any


def diagnostics_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)


def diagnostics_text(report: Mapping[str, Any]) -> str:
    lines = [
        f"LinaCore: {report['linacore_version']}",
        f"Python: {report['python_version']}",
        f"İşletim sistemi: {report['operating_system']} ({report['architecture']})",
        "Transformers: "
        f"{'kullanılabilir' if report['transformers_available'] else 'kullanılamıyor'}",
        f"Torch: {'kullanılabilir' if report['torch_available'] else 'kullanılamıyor'}",
        f"Device: {', '.join(report['devices'])}",
        f"Model sayısı: {report['discovered_model_count']}",
    ]
    lines.extend(f"Uyarı: {warning}" for warning in report["warnings"])
    return "\n".join(lines)
