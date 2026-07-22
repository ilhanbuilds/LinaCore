import importlib.util
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from lina_core.backends.transformers.availability import transformers_availability
from lina_core.core.version import __version__
from lina_core.diagnostics.hardware import detect_devices
from lina_core.models.discovery import discover_models

_PACKAGES = ("lina-core", "transformers", "torch")
_ENV_ALLOWLIST = ("LINA_CORE_LOG_LEVEL", "LINA_CORE_PROVIDER", "LINA_CORE_TOKENIZER")


def environment_report(model_root: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in _PACKAGES:
        if package == "lina-core":
            packages[package] = __version__
            continue
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    availability = transformers_availability()
    devices = detect_devices()
    discovery = discover_models(model_root)
    warnings = list(availability.warnings)
    if not availability.available:
        warnings.append(availability.reason)
    return {
        "linacore_version": __version__,
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "architecture": platform.machine(),
        "packages": packages,
        "transformers_available": availability.available,
        "torch_available": importlib.util.find_spec("torch") is not None,
        "cuda_available": any(item.device_type.value == "cuda" for item in devices),
        "devices": [item.identifier for item in devices],
        "model_root": model_root.name,
        "discovered_model_count": len(discovery.models),
        "model_root_writable": os.access(
            model_root if model_root.exists() else model_root.parent, os.W_OK
        ),
        "security": {
            "local_files_only_default": True,
            "trust_remote_code": False,
            "remote_downloads_default": False,
        },
        "environment": {name: "set" for name in _ENV_ALLOWLIST if name in os.environ},
        "warnings": warnings,
        "executable": Path(sys.executable).name,
    }
