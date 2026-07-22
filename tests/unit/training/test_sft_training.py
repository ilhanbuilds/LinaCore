from dataclasses import replace
from pathlib import Path

import pytest

from lina_core.core.exceptions import TrainingError
from lina_core.training.config import apply_training_profile, load_training_config
from lina_core.training.runner import TrainingRunner

CONFIG = Path("configs/training/lina-135m-sft-v0.1.toml")


def test_training_config_and_profiles_are_typed() -> None:
    config = load_training_config(CONFIG)
    assert config.local_files_only and not config.allow_download
    assert not config.trust_remote_code and config.report_to == "none"
    assert apply_training_profile(config, "smoke").max_steps == 1
    quick = apply_training_profile(config, "quick")
    assert quick.max_steps == 0 and quick.num_train_epochs == 3
    assert config.fingerprint.startswith("sha256:")


def test_training_plan_does_not_start_training() -> None:
    config = load_training_config(CONFIG)
    config = replace(config, output_dir="models/unit-test-training-plan")
    plan = TrainingRunner(Path.cwd()).plan(config)
    assert plan.record_count == 24
    assert not plan.download_required
    assert plan.method == "full-supervised-fine-tuning"


@pytest.mark.parametrize(
    "field,value",
    [
        ("trust_remote_code", True),
        ("report_to", "wandb"),
        ("dtype", "float16"),
        ("use_lora", True),
        ("train_split", -1.0),
    ],
)
def test_unsafe_training_config_is_rejected(field: str, value: object) -> None:
    config = load_training_config(CONFIG)
    with pytest.raises(TrainingError):
        replace(config, **{field: value})


def test_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    config = load_training_config(CONFIG)
    dataset = tmp_path / config.dataset_path
    dataset.parent.mkdir(parents=True)
    dataset.write_text(Path(config.dataset_path).read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "model"
    output.mkdir()
    (output / "existing").write_text("keep", encoding="utf-8")
    config = replace(config, output_dir=str(output.relative_to(tmp_path)))
    with pytest.raises(TrainingError, match="Output dizini dolu"):
        TrainingRunner(tmp_path).validate_output(config)
