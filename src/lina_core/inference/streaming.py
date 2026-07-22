from dataclasses import dataclass
from enum import StrEnum


class StreamEventType(StrEnum):
    STARTED = "started"
    TOKEN = "token"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    type: StreamEventType
    request_id: str
    text: str = ""
    index: int = 0
