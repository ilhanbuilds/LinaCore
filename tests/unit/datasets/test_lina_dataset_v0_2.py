import json
from pathlib import Path

from lina_core.datasets.quality import (
    audit_holdout_leakage,
    audit_sft_dataset,
    audit_split_leakage,
)
from lina_core.datasets.sft import load_sft_dataset, split_sft_dataset_grouped

SFT_PATH = Path("data/examples/lina_sft_v0.2.jsonl")
HOLDOUT_PATH = Path("data/examples/lina_behavior_holdout_v0.2.jsonl")


def _holdout_prompts() -> list[tuple[str, str]]:
    return [
        (item["id"], item["prompt"])
        for item in (
            json.loads(line) for line in HOLDOUT_PATH.read_text(encoding="utf-8").splitlines()
        )
    ]


def test_lina_dataset_v0_2_meets_size_balance_and_quality_gates() -> None:
    records = load_sft_dataset(SFT_PATH)
    report = audit_sft_dataset(records)

    assert len(records) == 600
    assert report.category_count == 30
    assert report.minimum_category_size == 20
    assert report.maximum_category_size == 20
    assert report.passed
    assert {issue.code for issue in report.issues} == {"cross-category-phrase-overuse"}


def test_lina_dataset_v0_2_has_leakage_safe_deterministic_split() -> None:
    records = load_sft_dataset(SFT_PATH)
    first = split_sft_dataset_grouped(
        records, seed=42, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1
    )
    second = split_sft_dataset_grouped(
        records, seed=42, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1
    )

    assert first == second
    assert (len(first.train), len(first.validation), len(first.test)) == (480, 60, 60)
    assert all(
        len({record.category for record in items}) == 30
        for items in (first.train, first.validation, first.test)
    )
    assert audit_split_leakage(first).passed


def test_lina_behavior_holdout_v0_2_is_large_and_isolated() -> None:
    prompts = _holdout_prompts()
    records = load_sft_dataset(SFT_PATH)

    assert len(prompts) == 90
    assert len({sample_id for sample_id, _ in prompts}) == 90
    assert audit_holdout_leakage(records, prompts).passed
