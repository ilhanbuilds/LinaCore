from typing import Generic, TypeVar

from lina_core.core.exceptions import ProviderNotFoundError

T = TypeVar("T")


class ProviderRegistry(Generic[T]):
    def __init__(self) -> None:
        self._providers: dict[str, T] = {}

    def register(self, provider_id: str, provider: T) -> None:
        if not provider_id:
            raise ValueError("Provider id cannot be empty")
        if provider_id in self._providers:
            raise ValueError(f"Provider already registered: {provider_id}")
        self._providers[provider_id] = provider

    def get(self, provider_id: str) -> T:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise ProviderNotFoundError(f"Provider not found: {provider_id}") from exc

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))
