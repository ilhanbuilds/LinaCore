from enum import StrEnum
from threading import RLock
from types import TracebackType

from lina_core.core.exceptions import LinaCoreError


class LifecycleState(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


_ALLOWED = {
    LifecycleState.CREATED: {LifecycleState.STARTING, LifecycleState.FAILED},
    LifecycleState.STARTING: {LifecycleState.READY, LifecycleState.FAILED},
    LifecycleState.READY: {LifecycleState.STOPPING, LifecycleState.FAILED},
    LifecycleState.STOPPING: {LifecycleState.STOPPED, LifecycleState.FAILED},
    LifecycleState.STOPPED: set(),
    LifecycleState.FAILED: {LifecycleState.STOPPING, LifecycleState.STOPPED},
}


class Lifecycle:
    def __init__(self) -> None:
        self._state = LifecycleState.CREATED
        self._lock = RLock()

    @property
    def state(self) -> LifecycleState:
        return self._state

    def transition(self, target: LifecycleState) -> None:
        with self._lock:
            if target not in _ALLOWED[self._state]:
                raise LinaCoreError(f"Invalid lifecycle transition: {self._state} -> {target}")
            self._state = target

    def start(self) -> None:
        self.transition(LifecycleState.STARTING)
        self.transition(LifecycleState.READY)

    def stop(self) -> None:
        if self.state is LifecycleState.READY:
            self.transition(LifecycleState.STOPPING)
        if self.state in {LifecycleState.STOPPING, LifecycleState.FAILED}:
            self.transition(LifecycleState.STOPPED)

    def __enter__(self) -> "Lifecycle":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc and self.state not in {LifecycleState.FAILED, LifecycleState.STOPPED}:
            self.transition(LifecycleState.FAILED)
        self.stop()
