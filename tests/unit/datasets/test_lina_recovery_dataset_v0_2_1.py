from pathlib import Path

from lina_core.datasets.quality import (
    assistant_phrase_frequencies,
    audit_dataset_unicode,
    audit_sft_dataset,
)
from lina_core.datasets.sft import load_sft_dataset
from lina_core.evaluation.behavior import load_behavior_samples
from lina_core.training.config import load_training_config
from scripts.generate_lina_recovery_dataset_v0_2_1 import (
    generate_holdout_records,
    generate_sft_records,
)

SFT_PATH = Path("data/examples/lina_recovery_canary_v0.2.1.jsonl")
HOLDOUT_PATH = Path("data/examples/lina_recovery_canary_holdout_v0.2.1.jsonl")
CONFIG_PATH = Path("configs/training/lina-0.5b-sft-v0.2.1-canary.toml")


def test_recovery_canary_dataset_is_small_balanced_and_deterministic() -> None:
    records = load_sft_dataset(SFT_PATH)

    assert len(records) == 32
    assert len(generate_sft_records()) == 32
    assert {record.category for record in records} == {
        "identity_and_scope",
        "safety_and_honesty",
    }
    assert audit_sft_dataset(records).passed
    assert not audit_sft_dataset(records).issues
    assert audit_dataset_unicode(records).passed
    assert not assistant_phrase_frequencies(
        records, ngram_size=5, minimum_count=3, minimum_categories=2
    )


def test_recovery_canary_holdout_is_independent_and_critical() -> None:
    samples = load_behavior_samples(HOLDOUT_PATH)

    assert len(samples) == len(generate_holdout_records()) == 8
    assert len({sample.prompt for sample in samples}) == 8
    assert sum(sample.critical for sample in samples) == 7


def test_recovery_canary_config_uses_lower_learning_rate_and_new_output() -> None:
    config = load_training_config(CONFIG_PATH)

    assert config.learning_rate == 0.00005
    assert config.max_steps == 48
    assert config.output_model_id == "lina-0.5b-sft-v0.2.1-canary"
    assert config.output_dir == "models/lina-0.5b-sft-v0.2.1-canary"
    assert config.local_files_only and not config.allow_download
