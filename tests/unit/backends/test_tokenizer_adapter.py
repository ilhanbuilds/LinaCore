from typing import Any, ClassVar

import pytest

from lina_core.backends.transformers.config import TransformersBackendConfig
from lina_core.backends.transformers.tokenizer import TransformersTokenizerProvider
from lina_core.core.exceptions import TokenizerError


class FakeTokenizer:
    chat_template: str | None = "template"
    model_max_length = 4096
    padding_side = "left"
    truncation_side = "right"
    special_tokens_map: ClassVar[dict[str, Any]] = {}

    def __len__(self) -> int:
        return 256

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(text.encode())

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return bytes(tokens).decode()


@pytest.fixture
def adapter() -> TransformersTokenizerProvider:
    return TransformersTokenizerProvider(TransformersBackendConfig("local"), FakeTokenizer())


@pytest.mark.parametrize("text", ["", "hello", "Türkçe", "emoji 💙", "two words"])
def test_adapter_round_trip(adapter: TransformersTokenizerProvider, text: str) -> None:
    assert adapter.decode(adapter.encode(text)) == text


@pytest.mark.parametrize("text", ["a", "abc", "Türkçe", "💙"])
def test_adapter_token_count(adapter: TransformersTokenizerProvider, text: str) -> None:
    assert adapter.count_tokens(text) == len(text.encode())


def test_adapter_metadata(adapter: TransformersTokenizerProvider) -> None:
    metadata = adapter.metadata
    assert metadata.vocabulary_size == 256
    assert metadata.model_max_length == 4096
    assert metadata.padding_side == "left"
    assert metadata.chat_template_available


def test_adapter_unload(adapter: TransformersTokenizerProvider) -> None:
    adapter.unload()
    with pytest.raises(TokenizerError):
        adapter.encode("x")


def test_adapter_requires_load_without_double() -> None:
    adapter = TransformersTokenizerProvider(TransformersBackendConfig("missing"))
    with pytest.raises(TokenizerError):
        _ = adapter.metadata


def test_adapter_invalid_token_sequence(adapter: TransformersTokenizerProvider) -> None:
    with pytest.raises((TokenizerError, ValueError)):
        adapter.decode([999])
