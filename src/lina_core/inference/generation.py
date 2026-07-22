from collections.abc import Iterable

from lina_core.inference.streaming import StreamEvent, StreamEventType


def validate_stream(events: Iterable[StreamEvent]) -> None:
    items = list(events)
    if not items or items[0].type is not StreamEventType.STARTED:
        raise ValueError("Stream must start with started")
    if items[-1].type not in {
        StreamEventType.COMPLETED,
        StreamEventType.CANCELLED,
        StreamEventType.ERROR,
    }:
        raise ValueError("Stream must terminate")
    if [e.index for e in items] != list(range(len(items))):
        raise ValueError("Stream indices must be ordered")
