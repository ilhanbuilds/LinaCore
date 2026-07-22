from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class TokenizerMetadata:
    tokenizer_id: str
    vocabulary_size: int
    special_tokens: Mapping[str, int]
    source: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "special_tokens", MappingProxyType(dict(self.special_tokens)))
