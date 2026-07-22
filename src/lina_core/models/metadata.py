from dataclasses import dataclass
from enum import StrEnum


class ModelCapability(StrEnum):
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    EMBEDDINGS = "embeddings"
    VISION = "vision"


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    model_id: str
    display_name: str
    version: str
    architecture: str
    context_length: int
    capabilities: frozenset[ModelCapability]
    source: str
    license: str
    parameter_count: int | None = None
    quantization: str | None = None
    loaded: bool = False
