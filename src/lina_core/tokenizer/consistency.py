import hashlib
import json
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenizerSignature:
    tokenizer_class: str
    length: int
    vocab_size: int
    chat_template_sha256: str
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    unk_token_id: int | None
    added_tokens: tuple[tuple[str, int], ...]
    special_tokens_sha256: str
    model_max_length: int
    padding_side: str
    truncation_side: str


@dataclass(frozen=True, slots=True)
class TokenizerParityReport:
    source: TokenizerSignature
    artifact: TokenizerSignature
    differences: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.differences


def tokenizer_signature(tokenizer: Any) -> TokenizerSignature:
    template = str(getattr(tokenizer, "chat_template", None) or "")
    special_tokens = json.dumps(
        getattr(tokenizer, "special_tokens_map", {}),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    added_tokens = tuple(
        sorted(
            (str(token), int(token_id))
            for token, token_id in getattr(tokenizer, "get_added_vocab", lambda: {})().items()
        )
    )
    return TokenizerSignature(
        type(tokenizer).__name__,
        len(tokenizer),
        int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        hashlib.sha256(template.encode("utf-8")).hexdigest(),
        _optional_int(getattr(tokenizer, "bos_token_id", None)),
        _optional_int(getattr(tokenizer, "eos_token_id", None)),
        _optional_int(getattr(tokenizer, "pad_token_id", None)),
        _optional_int(getattr(tokenizer, "unk_token_id", None)),
        added_tokens,
        hashlib.sha256(special_tokens.encode("utf-8")).hexdigest(),
        int(getattr(tokenizer, "model_max_length", 0)),
        str(getattr(tokenizer, "padding_side", "right")),
        str(getattr(tokenizer, "truncation_side", "right")),
    )


def compare_tokenizers(source: Any, artifact: Any) -> TokenizerParityReport:
    source_signature = tokenizer_signature(source)
    artifact_signature = tokenizer_signature(artifact)
    differences = tuple(
        item.name
        for item in fields(source_signature)
        if getattr(source_signature, item.name) != getattr(artifact_signature, item.name)
    )
    return TokenizerParityReport(source_signature, artifact_signature, differences)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | str):
        return int(value)
    raise TypeError("Tokenizer token ID değeri int veya str olmalıdır.")
