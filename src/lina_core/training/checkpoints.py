from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    checkpoint_id: str
    run_name: str
    step: int
    epoch: float
    path: Path
    checksum: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    resumable: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.step < 0 or self.epoch < 0:
            raise ValueError("Checkpoint position cannot be negative")
