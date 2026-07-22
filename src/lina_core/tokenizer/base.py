from abc import ABC, abstractmethod

from lina_core.tokenizer.metadata import TokenizerMetadata


class TokenizerProvider(ABC):
    @property
    @abstractmethod
    def metadata(self) -> TokenizerMetadata: ...
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...
    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...
    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def vocabulary_size(self) -> int:
        return self.metadata.vocabulary_size
