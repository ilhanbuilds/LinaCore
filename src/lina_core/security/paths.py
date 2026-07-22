from pathlib import Path

from lina_core.core.exceptions import SecurityError

FORBIDDEN_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_resolve(base: Path, candidate: str | Path, *, allow_absolute: bool = False) -> Path:
    base = base.resolve()
    item = Path(candidate)
    if item.is_absolute() and not allow_absolute:
        raise SecurityError("Absolute paths are not allowed")
    if any(part.upper().split(".")[0] in FORBIDDEN_NAMES for part in item.parts):
        raise SecurityError("Forbidden path name")
    resolved = (item if item.is_absolute() else base / item).resolve()
    if resolved != base and base not in resolved.parents:
        raise SecurityError("Path escapes the allowed workspace")
    return resolved
