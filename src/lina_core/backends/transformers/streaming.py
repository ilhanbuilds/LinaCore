from dataclasses import dataclass
from threading import Thread
from typing import Any


@dataclass(slots=True)
class StreamWorker:
    thread: Thread
    streamer: Any

    def close(self, timeout: float = 5.0) -> None:
        self.thread.join(timeout)
