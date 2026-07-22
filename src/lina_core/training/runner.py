import platform
import sys
from pathlib import Path
from typing import Any

from lina_core.core.exceptions import TrainingError
from lina_core.datasets.validation import load_manifest
from lina_core.security.paths import safe_resolve
from lina_core.training.config import TrainingConfig


class TrainingRunner:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def dry_run(self, config: TrainingConfig) -> dict[str, Any]:
        manifest = safe_resolve(self.workspace, config.dataset_manifest)
        output = safe_resolve(self.workspace, config.output_dir)
        load_manifest(manifest)
        if output.exists() and not output.is_dir():
            raise TrainingError("Output path is not a directory")
        return {
            "status": "dry-run-valid",
            "run_name": config.run_name,
            "model_id": config.model_id,
            "output_dir": str(output),
            "environment": {"python": sys.version.split()[0], "platform": platform.system()},
            "training_started": False,
        }
