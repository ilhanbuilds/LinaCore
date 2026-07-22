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
    model_max_length: int | None = None
    padding_side: str = "right"
    truncation_side: str = "right"
    chat_template_available: bool = False
    chat_template_fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "special_tokens", MappingProxyType(dict(self.special_tokens)))
