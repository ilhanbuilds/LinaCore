import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lina_core.core.exceptions import DatasetError
from lina_core.inference.request import ChatMessage, ChatRole
from lina_core.security.limits import ResourceLimits
from lina_core.security.secrets import contains_secret

_ALLOWED_FIELDS = {
    "id",
    "messages",
    "category",
    "language",
    "source",
    "license",
    "quality_status",
    "metadata",
}
_ALLOWED_MESSAGE_FIELDS = {"role", "content"}
_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
_LOCAL_PATH = re.compile(r"(?i)(?:[A-Z]:[\\/](?:Users|Desktop|Documents)[\\/]|/home/|/Users/)")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{8,}\d)(?!\d)")


@dataclass(frozen=True, slots=True)
class SFTRecord:
    record_id: str
    messages: tuple[ChatMessage, ...]
    category: str
    language: str
    source: str
    license: str
    quality_status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "messages": [
                {"role": message.role.value, "content": _normalize(message.content)}
                for message in self.messages
            ],
            "category": self.category,
            "language": self.language,
            "source": self.source,
            "license": self.license,
            "quality_status": self.quality_status,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class SFTDatasetReport:
    path: str
    record_count: int
    valid_count: int
    invalid_count: int
    fingerprint: str
    languages: dict[str, int]
    categories: dict[str, int]
    message_characters: dict[str, float]
    approximate_tokens: int


@dataclass(frozen=True, slots=True)
class SFTDatasetSplit:
    train: tuple[SFTRecord, ...]
    validation: tuple[SFTRecord, ...]
    test: tuple[SFTRecord, ...]


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n")).strip()


def _parse_record(data: Any, line_number: int) -> SFTRecord:
    if not isinstance(data, dict):
        raise DatasetError(f"Satır {line_number}: kayıt object olmalıdır.")
    unknown = set(data) - _ALLOWED_FIELDS
    missing = _ALLOWED_FIELDS - set(data)
    if unknown or missing:
        raise DatasetError(
            f"Satır {line_number}: bilinmeyen={sorted(unknown)}, eksik={sorted(missing)}"
        )
    raw_messages = data["messages"]
    if not isinstance(raw_messages, list) or not raw_messages:
        raise DatasetError(f"Satır {line_number}: messages boş olamaz.")
    messages: list[ChatMessage] = []
    for raw in raw_messages:
        if not isinstance(raw, dict) or set(raw) != _ALLOWED_MESSAGE_FIELDS:
            raise DatasetError(f"Satır {line_number}: geçersiz message alanları.")
        try:
            role = ChatRole(raw["role"])
            content = _normalize(str(raw["content"]))
            messages.append(ChatMessage(role, content))
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"Satır {line_number}: geçersiz mesaj: {exc}") from exc
    record = SFTRecord(
        str(data["id"]),
        tuple(messages),
        str(data["category"]),
        str(data["language"]),
        str(data["source"]),
        str(data["license"]),
        str(data["quality_status"]),
        dict(data["metadata"]) if isinstance(data["metadata"], dict) else {},
    )
    validate_sft_record(record)
    return record


def validate_sft_record(record: SFTRecord) -> None:
    if not record.record_id or not record.category or not record.source or not record.license:
        raise DatasetError("Kimlik, kategori, kaynak ve lisans zorunludur.")
    if record.quality_status not in {"approved", "reviewed"}:
        raise DatasetError("Kayıt eğitim için onaylanmamış.")
    if not _LANGUAGE.fullmatch(record.language):
        raise DatasetError("Geçersiz dil kodu.")
    if not record.messages or record.messages[-1].role is not ChatRole.ASSISTANT:
        raise DatasetError("Kayıt assistant cevabıyla bitmelidir.")
    combined = "\n".join(message.content for message in record.messages)
    if not all(message.content for message in record.messages):
        raise DatasetError("Boş mesaj kabul edilmez.")
    if len(combined.encode("utf-8")) > ResourceLimits().max_dataset_record_size:
        raise DatasetError("Kayıt boyut sınırını aşıyor.")
    if contains_secret(combined) or _LOCAL_PATH.search(combined):
        raise DatasetError("Kayıt secret veya kişisel dosya yolu içeriyor.")
    if _EMAIL.search(combined) or _PHONE.search(combined):
        raise DatasetError("Kayıt kişisel veri benzeri içerik taşıyor.")


def load_sft_dataset(path: Path) -> list[SFTRecord]:
    records: list[SFTRecord] = []
    ids: set[str] = set()
    contents: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Dataset okunamadı: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Satır {line_number}: geçersiz JSON: {exc}") from exc
        record = _parse_record(raw, line_number)
        content_key = json.dumps(record.canonical()["messages"], ensure_ascii=False, sort_keys=True)
        if record.record_id in ids:
            raise DatasetError(f"Yinelenen kayıt kimliği: {record.record_id}")
        if content_key in contents:
            raise DatasetError(f"Yinelenen mesaj içeriği: {record.record_id}")
        ids.add(record.record_id)
        contents.add(content_key)
        records.append(record)
    if not records:
        raise DatasetError("Dataset kayıt içermiyor.")
    return records


def dataset_fingerprint(records: list[SFTRecord] | tuple[SFTRecord, ...]) -> str:
    payload = "\n".join(
        json.dumps(item.canonical(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in sorted(records, key=lambda item: item.record_id)
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_sft_dataset(
    records: list[SFTRecord],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> SFTDatasetSplit:
    if (
        min(train_ratio, validation_ratio, test_ratio) < 0
        or abs(train_ratio + validation_ratio + test_ratio - 1) > 1e-9
    ):
        raise DatasetError("Split oranları negatif olmamalı ve toplamı 1 olmalıdır.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    validation_count = max(1, round(count * validation_ratio)) if validation_ratio else 0
    test_count = max(1, round(count * test_ratio)) if test_ratio else 0
    if validation_count + test_count >= count:
        validation_count = 1 if count >= 3 and validation_ratio else 0
        test_count = 1 if count >= 3 and test_ratio else 0
    train_count = count - validation_count - test_count
    if train_count <= 0:
        raise DatasetError("Dataset anlamlı bir train split için çok küçük.")
    return SFTDatasetSplit(
        tuple(shuffled[:train_count]),
        tuple(shuffled[train_count : train_count + validation_count]),
        tuple(shuffled[train_count + validation_count :]),
    )


def split_sft_dataset_grouped(
    records: list[SFTRecord],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> SFTDatasetSplit:
    """Split by category while keeping related template groups together."""
    _validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    categories: dict[str, list[SFTRecord]] = defaultdict(list)
    for record in records:
        categories[record.category].append(record)

    split_items: dict[str, list[SFTRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for category, category_records in sorted(categories.items()):
        groups: dict[str, list[SFTRecord]] = defaultdict(list)
        for record in category_records:
            group = str(record.metadata.get("template_group") or record.record_id)
            groups[group].append(record)
        ordered_groups = sorted(groups.items())
        random.Random(f"{seed}:{category}").shuffle(ordered_groups)
        validation_groups = max(1, round(len(ordered_groups) * validation_ratio))
        test_groups = max(1, round(len(ordered_groups) * test_ratio))
        if validation_groups + test_groups >= len(ordered_groups):
            raise DatasetError(
                f"{category} kategorisi grouped split için yeterli template grubu içermiyor."
            )
        train_groups = len(ordered_groups) - validation_groups - test_groups
        destinations = (
            ("train", ordered_groups[:train_groups]),
            (
                "validation",
                ordered_groups[train_groups : train_groups + validation_groups],
            ),
            ("test", ordered_groups[train_groups + validation_groups :]),
        )
        for destination, selected_groups in destinations:
            for _, group_records in selected_groups:
                split_items[destination].extend(group_records)

    for destination, items in split_items.items():
        random.Random(f"{seed}:{destination}").shuffle(items)
    return SFTDatasetSplit(
        tuple(split_items["train"]),
        tuple(split_items["validation"]),
        tuple(split_items["test"]),
    )


def _validate_split_ratios(
    train_ratio: float, validation_ratio: float, test_ratio: float
) -> None:
    if (
        min(train_ratio, validation_ratio, test_ratio) <= 0
        or abs(train_ratio + validation_ratio + test_ratio - 1) > 1e-9
    ):
        raise DatasetError("Grouped split oranları pozitif olmalı ve toplamı 1 olmalıdır.")


def dataset_report(path: Path, records: list[SFTRecord]) -> SFTDatasetReport:
    lengths = [len(message.content) for item in records for message in item.messages]
    total_characters = sum(lengths)
    return SFTDatasetReport(
        path.name,
        len(records),
        len(records),
        0,
        dataset_fingerprint(records),
        dict(Counter(item.language for item in records)),
        dict(Counter(item.category for item in records)),
        {
            "minimum": float(min(lengths)),
            "maximum": float(max(lengths)),
            "average": total_characters / len(lengths),
        },
        max(1, round(total_characters / 4)),
    )


def render_record(record: SFTRecord) -> str:
    labels = {ChatRole.SYSTEM: "Sistem", ChatRole.USER: "Kullanıcı", ChatRole.ASSISTANT: "Lina"}
    return "\n".join(f"{labels[message.role]}: {message.content}" for message in record.messages)
