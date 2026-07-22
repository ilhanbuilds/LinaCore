from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from lina_core.core.exceptions import InferenceError
from lina_core.inference.sampling import GenerationOptions
from lina_core.security.limits import ResourceLimits
from lina_core.security.secrets import contains_secret


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise InferenceError("Message content cannot be empty")
        if len(self.content) > ResourceLimits().max_prompt_characters:
            raise InferenceError("Message is too large")
        if len(self.metadata) > 16 or any(
            not isinstance(k, str) or len(k) > 64 for k in self.metadata
        ):
            raise InferenceError("Unsafe metadata")
        if any(contains_secret(str(value)) for value in self.metadata.values()):
            raise InferenceError("Secrets are not allowed in metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _validate_order(messages: tuple[ChatMessage, ...]) -> None:
    if sum(m.role is ChatRole.SYSTEM for m in messages) > 1:
        raise InferenceError("Only one system message is allowed")
    if any(m.role is ChatRole.SYSTEM for m in messages[1:]):
        raise InferenceError("System message must be first")
    conversational = [m.role for m in messages if m.role is not ChatRole.SYSTEM]
    if not conversational or conversational[0] is not ChatRole.USER:
        raise InferenceError("Conversation must begin with a user message")
    if conversational[-1] is not ChatRole.USER:
        raise InferenceError("Conversation must end with a user message")


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    messages: tuple[ChatMessage, ...]
    options: GenerationOptions = field(default_factory=GenerationOptions)
    request_id: str = field(default_factory=lambda: str(uuid4()))
    model_id: str | None = None
    stream: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.messages:
            raise InferenceError("At least one message is required")
        if len(self.messages) > ResourceLimits().max_messages:
            raise InferenceError("Too many messages")
        if sum(len(m.content) for m in self.messages) > ResourceLimits().max_prompt_characters:
            raise InferenceError("Prompt is too large")
        _validate_order(self.messages)
