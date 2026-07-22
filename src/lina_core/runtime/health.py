from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class HealthLevel(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class HealthStatus:
    status: HealthLevel
    message: str
    details: Mapping[str, Any]
