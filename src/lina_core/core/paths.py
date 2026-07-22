from dataclasses import dataclass
from pathlib import Path

from lina_core.security.paths import safe_resolve


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        current = (start or Path.cwd()).resolve()
        for item in (current, *current.parents):
            if (item / "pyproject.toml").exists():
                return cls(item)
        return cls(current)

    @property
    def config_dir(self) -> Path:
        return self.root / "configs"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def model_dir(self) -> Path:
        return self.root / "models"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def report_dir(self) -> Path:
        return self.root / "reports"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    def resolve(self, candidate: str | Path) -> Path:
        return safe_resolve(self.root, candidate)
