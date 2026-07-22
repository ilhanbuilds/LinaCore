import importlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResourceReport:
    process_memory_bytes: int | None = None
    model_load_seconds: float | None = None
    model_memory_estimate: int | None = None
    active_requests: int = 0
    generated_tokens: int = 0
    last_inference_latency: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def process_memory_bytes() -> int | None:
    try:
        resource: Any = importlib.import_module("resource")
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if value > 10_000_000 else value * 1024)
    except (AttributeError, ImportError, OSError):
        return None
