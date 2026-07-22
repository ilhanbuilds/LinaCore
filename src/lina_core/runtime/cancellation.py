from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Event, RLock


@dataclass(frozen=True, slots=True)
class CancellationState:
    request_id: str
    cancelled: bool
    reason: str | None = None
    requested_at: datetime | None = None


@dataclass(slots=True)
class CancellationToken:
    request_id: str
    _event: Event = field(default_factory=Event, repr=False)
    _reason: str | None = field(default=None, repr=False)
    _requested_at: datetime | None = field(default=None, repr=False)

    def cancel(self, reason: str = "user") -> bool:
        if self._event.is_set():
            return False
        self._reason = reason
        self._requested_at = datetime.now(UTC)
        self._event.set()
        return True

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def state(self) -> CancellationState:
        return CancellationState(self.request_id, self.cancelled, self._reason, self._requested_at)


class CancellationRegistry:
    def __init__(self) -> None:
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = RLock()

    def create(self, request_id: str) -> CancellationToken:
        with self._lock:
            token = self._tokens.get(request_id)
            if token is None:
                token = CancellationToken(request_id)
                self._tokens[request_id] = token
            return token

    def cancel(self, request_id: str, reason: str = "user") -> bool:
        with self._lock:
            token = self._tokens.get(request_id)
            return token.cancel(reason) if token else False

    def finish(self, request_id: str) -> None:
        with self._lock:
            self._tokens.pop(request_id, None)

    def cancel_all(self, reason: str = "shutdown") -> int:
        with self._lock:
            return sum(token.cancel(reason) for token in self._tokens.values())

    @property
    def active_count(self) -> int:
        return len(self._tokens)
