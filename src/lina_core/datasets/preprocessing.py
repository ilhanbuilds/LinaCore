import re
import unicodedata
from dataclasses import dataclass

from lina_core.datasets.contracts import DatasetRecord


def normalize_text(value: str) -> str:
    return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFC", value.replace("\r\n", "\n"))).strip()


def fingerprint(record: DatasetRecord) -> str:
    import hashlib

    return hashlib.sha256(normalize_text(record.output).casefold().encode()).hexdigest()


def quality_score(record: DatasetRecord) -> float:
    length = len(normalize_text(record.output))
    return min(1.0, length / 80) if length else 0.0


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        if (
            abs(self.train + self.validation + self.test - 1) > 1e-9
            or min(self.train, self.validation, self.test) < 0
        ):
            raise ValueError("Splits must be non-negative and sum to one")
