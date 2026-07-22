import importlib.util
from importlib.metadata import PackageNotFoundError, version

from lina_core.backends.base import BackendAvailability
from lina_core.diagnostics.hardware import detect_devices


def transformers_availability() -> BackendAvailability:
    required = ("transformers", "torch")
    installed: dict[str, str] = {}
    missing: list[str] = []
    for package in required:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
            continue
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = "unknown"
    available = not missing
    reason = (
        "Transformers backend kullanıma hazır."
        if available
        else "Transformers backend kullanılamıyor. "
        "Gerekli optional dependency grubunu yükleyin: .[inference-transformers,inference-torch]"
    )
    return BackendAvailability(
        available,
        "transformers",
        installed,
        tuple(missing),
        installed.get("transformers"),
        reason,
        detect_devices(),
    )
