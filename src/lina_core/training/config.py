import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from lina_core.core.exceptions import TrainingError


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run_name: str
    output_dir: str
    base_model: str = ""
    base_revision: str = ""
    dataset_path: str = ""
    model_id: str = ""
    output_model_id: str = "lina-135m-sft-v0.1"
    display_name: str = "Lina-135M-SFT-v0.1"
    model_version: str = "0.1"
    dataset_version: str = "0.1"
    base_model_id: str = "HuggingFaceTB/SmolLM2-135M"
    base_license: str = "Apache-2.0"
    architecture: str = "SmolLM2ForCausalLM"
    parameter_count: int = 135_000_000
    tokenizer_id: str = ""
    dataset_manifest: str = ""
    dataset_format: str = "jsonl"
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    max_sequence_length: int = 256
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    split_strategy: str = "random"
    num_train_epochs: int = 1
    max_steps: int = 0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 1
    evaluation_steps: int = 1
    save_steps: int = 10
    save_total_limit: int = 2
    gradient_checkpointing: bool = False
    use_lora: bool = False
    merge_adapter: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ()
    early_stopping_patience: int = 0
    report_to: str = "none"
    allow_download: bool = False
    trust_remote_code: bool = False
    local_files_only: bool = True
    overwrite_output: bool = False
    resume_from_checkpoint: str | None = None
    epochs: int | None = None
    precision: str | None = None
    gradient_accumulation: int | None = None
    checkpoint_interval: int | None = None
    evaluation_interval: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.lora_target_modules, list):
            object.__setattr__(self, "lora_target_modules", tuple(self.lora_target_modules))
        if not self.base_model and self.model_id:
            object.__setattr__(self, "base_model", self.model_id)
        if not self.base_revision and self.model_id:
            object.__setattr__(self, "base_revision", "unversioned")
        if not self.dataset_path and self.dataset_manifest:
            object.__setattr__(self, "dataset_path", self.dataset_manifest)
        if self.epochs is not None:
            object.__setattr__(self, "num_train_epochs", self.epochs)
        if self.precision is not None:
            try:
                object.__setattr__(
                    self,
                    "dtype",
                    {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[self.precision],
                )
            except KeyError as exc:
                raise TrainingError("Desteklenmeyen precision.") from exc
        if self.gradient_accumulation is not None:
            object.__setattr__(self, "gradient_accumulation_steps", self.gradient_accumulation)
        if self.checkpoint_interval is not None:
            object.__setattr__(self, "save_steps", self.checkpoint_interval)
        if self.evaluation_interval is not None:
            object.__setattr__(self, "evaluation_steps", self.evaluation_interval)
        required = (
            self.run_name,
            self.output_dir,
            self.base_model,
            self.base_revision,
            self.dataset_path,
        )
        if not all(value.strip() for value in required):
            raise TrainingError("Zorunlu training config değeri boş olamaz.")
        if self.dataset_format != "jsonl" and not self.dataset_manifest:
            raise TrainingError("Yalnız jsonl SFT dataset formatı destekleniyor.")
        if self.device not in {"cpu", "cuda", "cuda:0"}:
            raise TrainingError("Desteklenmeyen training device.")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise TrainingError("Desteklenmeyen training dtype.")
        if self.device == "cpu" and self.dtype == "float16":
            raise TrainingError("CPU üzerinde float16 training desteklenmiyor.")
        if (
            min(self.train_split, self.validation_split, self.test_split) < 0
            or abs(self.train_split + self.validation_split + self.test_split - 1) > 1e-9
        ):
            raise TrainingError("Training split oranlarının toplamı 1 olmalıdır.")
        positive = (
            self.num_train_epochs,
            self.max_sequence_length,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.logging_steps,
            self.evaluation_steps,
            self.save_steps,
            self.save_total_limit,
        )
        if min(positive) <= 0 or self.max_steps < 0:
            raise TrainingError("Training sayısal değerleri geçersiz.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.warmup_ratio <= 1:
            raise TrainingError("Optimizer ayarları geçersiz.")
        if self.trust_remote_code:
            raise TrainingError("trust_remote_code güvenlik politikası gereği reddedildi.")
        if not self.local_files_only and not self.allow_download:
            raise TrainingError("Remote model erişimi için allow_download açık olmalıdır.")
        if self.report_to != "none":
            raise TrainingError("Harici training telemetry bu sürümde desteklenmiyor.")
        if self.split_strategy not in {"random", "grouped"}:
            raise TrainingError("Desteklenmeyen dataset split stratejisi.")
        if self.merge_adapter and not self.use_lora:
            raise TrainingError("Adapter birleştirme yalnız LoRA ile kullanılabilir.")
        if self.use_lora and (
            self.lora_r <= 0 or self.lora_alpha <= 0 or not 0 <= self.lora_dropout < 1
        ):
            raise TrainingError("LoRA ayarları geçersiz.")
        if self.parameter_count <= 0 or self.early_stopping_patience < 0:
            raise TrainingError("Model metadata veya early stopping ayarı geçersiz.")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def apply_training_profile(config: TrainingConfig, profile: str) -> TrainingConfig:
    if profile == "smoke":
        return replace(config, max_steps=1, num_train_epochs=1, max_sequence_length=96)
    if profile == "quick":
        return replace(config, max_steps=0, num_train_epochs=3, max_sequence_length=128)
    if profile == "quality":
        return replace(
            config,
            num_train_epochs=max(3, config.num_train_epochs),
            max_sequence_length=max(256, config.max_sequence_length),
        )
    raise TrainingError(f"Bilinmeyen training profili: {profile}")


def load_training_config(path: Path) -> TrainingConfig:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        section = data.get("training", data)
        unknown = set(section) - {item.name for item in fields(TrainingConfig)}
        if unknown:
            raise TrainingError(
                f"Bilinmeyen training config alanları: {', '.join(sorted(unknown))}"
            )
        return TrainingConfig(**section)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, TypeError) as exc:
        raise TrainingError(f"Geçersiz training config: {exc}") from exc
