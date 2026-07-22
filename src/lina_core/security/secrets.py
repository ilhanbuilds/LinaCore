import re
from collections.abc import Mapping
from typing import Any

_SECRET = re.compile(r"(?i)(bearer\s+[a-z0-9._-]{12,}|(?:api[_-]?key|token|password|secret|credential)\s*[:=]\s*\S+|-----BEGIN [A-Z ]*PRIVATE KEY-----)")

def contains_secret(value: str) -> bool:
    return bool(_SECRET.search(value))

def redact(value: str) -> str:
    return _SECRET.sub("[REDACTED]", value)

def redact_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: "[REDACTED]" if re.search(r"(?i)token|secret|password|credential|key", key) else val for key, val in values.items()}

