from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ApiResponse:
    ok: bool
    data: Mapping[str, Any] | None = None
    error: str | None = None
