from collections import Counter
from pathlib import Path

from lina_core.datasets.quality import (
    assistant_phrase_frequencies,
    audit_dataset_unicode,
    audit_holdout_leakage,
    audit_sft_dataset,
)
from lina_core.datasets.sft import dataset_fingerprint, load_sft_dataset
from lina_core.evaluation.behavior import load_behavior_samples
from lina_core.training.config import load_training_config
from scripts.generate_lina_recovery_dataset_v0_3 import (
    generate_holdout_records,
    generate_sft_records,
    generate_stage_records,
)

DATASET_PATH = Path("data/examples/lina_recovery_v0.3.jsonl")
HOLDOUT_PATH = Path("data/evaluation/lina_recovery_holdout_v0.3.jsonl")


def test_curated_recovery_dataset_is_balanced_deterministic_and_clean() -> None:
    records = load_sft_dataset(DATASET_PATH)

    assert len(records) == len(generate_sft_records()) == 96
    assert set(Counter(record.category for record in records).values()) == {6}
    assert len({record.category for record in records}) == 16
    assert (
        dataset_fingerprint(records)
        == "sha256:6c48fde12484a4e17d3ea0c6b409aa37511e68c40095d4f45fc0166c1a205354"
    )
    assert audit_sft_dataset(records, minimum_category_size=6).passed
    assert audit_dataset_unicode(records).passed
    assert not assistant_phrase_frequencies(
        records, ngram_size=5, minimum_count=3, minimum_categories=2
    )


def test_recovery_holdout_is_independent_and_leakage_safe() -> None:
    records = load_sft_dataset(DATASET_PATH)
    samples = load_behavior_samples(HOLDOUT_PATH)

    assert len(samples) == len(generate_holdout_records()) == 32
    assert len({sample.prompt for sample in samples}) == 32
    assert sum(sample.critical for sample in samples) == 16
    assert audit_holdout_leakage(
        records, ((sample.sample_id, sample.prompt) for sample in samples)
    ).passed


def test_curriculum_subsets_are_deterministic_and_bounded() -> None:
    generated = generate_sft_records()
    stage_one = load_sft_dataset(Path("data/examples/lina_recovery_v0.3_stage1.jsonl"))
    stage_two = load_sft_dataset(Path("data/examples/lina_recovery_v0.3_stage2.jsonl"))

    assert len(stage_one) == len(generate_stage_records(generated, 1)) == 24
    assert len(stage_two) == len(generate_stage_records(generated, 2)) == 64
    assert {record.category for record in stage_one} == {
        "identity",
        "secret_safety",
        "file_access_honesty",
        "missing_information",
    }


def test_curriculum_configs_are_local_bounded_and_conservative() -> None:
    stage_one = load_training_config(
        Path("configs/training/lina-0.5b-sft-v0.2.1-natural-stage1.toml")
    )
    stage_two = load_training_config(
        Path("configs/training/lina-0.5b-sft-v0.2.1-natural-stage2.toml")
    )

    assert stage_one.learning_rate == stage_two.learning_rate == 0.000075
    assert stage_one.max_steps == 48
    assert stage_two.max_steps == 64
    assert stage_one.lora_r == stage_two.lora_r == 8
    assert stage_one.local_files_only and not stage_one.allow_download
    assert stage_two.local_files_only and not stage_two.allow_download
    assert stage_one.output_dir != stage_two.output_dir
