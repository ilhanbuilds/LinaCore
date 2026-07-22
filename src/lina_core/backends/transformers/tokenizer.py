from pathlib import Path
from typing import Any

from lina_core.backends.transformers.config import TransformersBackendConfig
from lina_core.backends.transformers.loader import load_modules
from lina_core.core.exceptions import TokenizerError
from lina_core.tokenizer.base import TokenizerProvider
from lina_core.tokenizer.metadata import TokenizerMetadata


class TransformersTokenizerProvider(TokenizerProvider):
    def __init__(self, config: TransformersBackendConfig, tokenizer: Any | None = None) -> None:
        self.config = config
        self._tokenizer = tokenizer
        self._fallback_used = False

    def load(self) -> None:
        if self._tokenizer is not None:
            return
        transformers, _ = load_modules()
        source = self.config.tokenizer_path or self.config.model_path
        if self.config.local_files_only and not Path(source).exists():
            raise TokenizerError("Yerel tokenizer yolu bulunamadı.")
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                source,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
                use_fast=True,
            )
        except (OSError, ValueError) as exc:
            raise TokenizerError(f"Tokenizer yüklenemedi: {exc}") from exc

    @property
    def raw_tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise TokenizerError("Tokenizer yüklenmedi.")
        return self._tokenizer

    @property
    def metadata(self) -> TokenizerMetadata:
        tokenizer = self.raw_tokenizer
        special = {
            str(key): int(value)
            for key, value in getattr(tokenizer, "special_tokens_map", {}).items()
            if isinstance(value, int)
        }
        return TokenizerMetadata(
            f"{self.config.model_id}-tokenizer",
            len(tokenizer),
            special,
            self.config.tokenizer_path or self.config.model_path,
            "transformers",
            getattr(tokenizer, "model_max_length", None),
            getattr(tokenizer, "padding_side", "right"),
            getattr(tokenizer, "truncation_side", "right"),
            bool(getattr(tokenizer, "chat_template", None)),
            self._fallback_used,
        )

    def encode(self, text: str) -> list[int]:
        try:
            return list(self.raw_tokenizer.encode(text, add_special_tokens=False))
        except (TypeError, ValueError) as exc:
            raise TokenizerError(f"Metin tokenize edilemedi: {exc}") from exc

    def decode(self, tokens: list[int]) -> str:
        try:
            return str(self.raw_tokenizer.decode(tokens, skip_special_tokens=True))
        except (TypeError, ValueError) as exc:
            raise TokenizerError(f"Tokenlar decode edilemedi: {exc}") from exc

    def unload(self) -> None:
        self._tokenizer = None
