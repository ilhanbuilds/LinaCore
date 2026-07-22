import json
from dataclasses import asdict
from pathlib import Path

from lina_core.evaluation.models import EvaluationResult
from lina_core.security.limits import ResourceLimits
from lina_core.security.paths import safe_resolve


def write_report(result: EvaluationResult, report_dir: Path, name: str) -> Path:
    path = safe_resolve(report_dir, name)
    payload = json.dumps(asdict(result), default=str, ensure_ascii=False, indent=2)
    if len(payload.encode()) > ResourceLimits().max_report_size:
        raise ValueError("Report too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return path
