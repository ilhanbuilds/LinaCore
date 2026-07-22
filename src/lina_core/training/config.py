import tomllib
from dataclasses import dataclass
from pathlib import Path

from lina_core.core.exceptions import TrainingError


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run_name: str
    model_id: str
    tokenizer_id: str
    dataset_manifest: str
    output_dir: str
    seed: int = 42
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 0.0001
    warmup_ratio: float = 0.0
    max_sequence_length: int = 1024
    precision: str = "fp32"
    checkpoint_interval: int = 100
    evaluation_interval: int = 100
    save_total_limit: int = 2

    def __post_init__(self) -> None:
        if not all(
            (
                self.run_name,
                self.model_id,
                self.tokenizer_id,
                self.dataset_manifest,
                self.output_dir,
            )
        ):
            raise TrainingError("Required training value is empty")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.gradient_accumulation,
                self.max_sequence_length,
                self.checkpoint_interval,
                self.evaluation_interval,
                self.save_total_limit,
            )
            <= 0
        ):
            raise TrainingError("Training numeric values must be positive")
        if self.learning_rate <= 0 or not 0 <= self.warmup_ratio <= 1:
            raise TrainingError("Invalid learning rate or warmup ratio")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise TrainingError("Unsupported precision")


def load_training_config(path: Path) -> TrainingConfig:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return TrainingConfig(**data.get("training", data))
    except (OSError, tomllib.TOMLDecodeError, TypeError) as exc:
        raise TrainingError(f"Invalid training config: {exc}") from exc
