import json
from pathlib import Path

import pytest

from lina_core.core.exceptions import DatasetError
from lina_core.datasets.sft import (
    dataset_fingerprint,
    dataset_report,
    load_sft_dataset,
    render_record,
    split_sft_dataset,
    split_sft_dataset_grouped,
)


def _record(record_id: str, content: str = "Merhaba") -> dict[str, object]:
    return {
        "id": record_id,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": f"Yanıt {record_id}"},
        ],
        "category": "test",
        "language": "tr",
        "source": "synthetic",
        "license": "CC0-1.0",
        "quality_status": "approved",
        "metadata": {},
    }


def _write(path: Path, items: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )


def test_example_dataset_is_valid() -> None:
    path = Path("data/examples/lina_sft_v0.1.jsonl")
    records = load_sft_dataset(path)
    report = dataset_report(path, records)
    assert report.record_count == 24
    assert len(report.categories) >= 20
    assert report.fingerprint.startswith("sha256:")
    assert "Lina:" in render_record(records[0])


def test_fingerprint_is_order_independent(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    _write(path, [_record("a", "one"), _record("b", "two")])
    records = load_sft_dataset(path)
    assert dataset_fingerprint(records) == dataset_fingerprint(list(reversed(records)))


def test_split_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    _write(path, [_record(str(index), str(index)) for index in range(10)])
    records = load_sft_dataset(path)
    first = split_sft_dataset(
        records, seed=42, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1
    )
    second = split_sft_dataset(
        records, seed=42, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1
    )
    assert first == second
    assert (len(first.train), len(first.validation), len(first.test)) == (8, 1, 1)


def test_grouped_split_preserves_categories_and_template_groups(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    items: list[dict[str, object]] = []
    for category in ("identity", "privacy"):
        for group_index in range(10):
            for variant in range(2):
                item = _record(
                    f"{category}-{group_index}-{variant}",
                    f"{category} prompt {group_index} variant {variant}",
                )
                item["category"] = category
                item["metadata"] = {"template_group": f"{category}-{group_index}"}
                items.append(item)
    _write(path, items)
    split = split_sft_dataset_grouped(
        load_sft_dataset(path),
        seed=42,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )

    assert (len(split.train), len(split.validation), len(split.test)) == (32, 4, 4)
    for records in (split.train, split.validation, split.test):
        assert {record.category for record in records} == {"identity", "privacy"}
    group_destinations: dict[str, set[str]] = {}
    for destination, records in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for record in records:
            group = str(record.metadata["template_group"])
            group_destinations.setdefault(group, set()).add(destination)
    assert all(len(destinations) == 1 for destinations in group_destinations.values())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update(extra=True),
        lambda item: item.update(messages=[]),
        lambda item: item.update(quality_status="draft"),
        lambda item: item.update(language="turkish"),
        lambda item: item["messages"][0].update(content="C:/Users/example/private"),
        lambda item: item["messages"][0].update(content="person@example.com"),
    ],
)
def test_invalid_records_are_rejected(tmp_path: Path, mutation: object) -> None:
    path = tmp_path / "dataset.jsonl"
    item = _record("a")
    mutation(item)  # type: ignore[operator]
    _write(path, [item])
    with pytest.raises(DatasetError):
        load_sft_dataset(path)


def test_duplicate_id_and_content_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    _write(path, [_record("a"), _record("a", "different")])
    with pytest.raises(DatasetError, match="Yinelenen kayıt"):
        load_sft_dataset(path)
