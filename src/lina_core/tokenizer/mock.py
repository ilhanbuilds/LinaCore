from lina_core.core.exceptions import TokenizerError
from lina_core.tokenizer.base import TokenizerProvider
from lina_core.tokenizer.metadata import TokenizerMetadata


class MockTokenizerProvider(TokenizerProvider):
    """Reversible UTF-8 byte tokenizer for architecture tests only."""

    @property
    def metadata(self) -> TokenizerMetadata:
        return TokenizerMetadata("mock", 256, {}, "built-in", "1")

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        if any(not 0 <= token <= 255 for token in tokens):
            raise TokenizerError("Invalid mock token")
        try:
            return bytes(tokens).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TokenizerError("Invalid UTF-8 token sequence") from exc
