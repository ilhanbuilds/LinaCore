from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lina_core.inference.request import ChatMessage


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    record_id: str
    output: str
    source: str
    language: str
    instruction: str | None = None
    input: str | None = None
    messages: tuple[ChatMessage, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    version: str
    split: str
    record_count: int
    language: str
    license: str
    checksum: str
    source: str
