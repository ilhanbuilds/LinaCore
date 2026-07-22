import importlib
from typing import Any

from lina_core.backends.errors import BackendUnavailableError
from lina_core.backends.transformers.availability import transformers_availability


def load_modules() -> tuple[Any, Any]:
    availability = transformers_availability()
    if not availability.available:
        raise BackendUnavailableError(availability.reason)
    try:
        return importlib.import_module("transformers"), importlib.import_module("torch")
    except ImportError as exc:
        raise BackendUnavailableError(availability.reason) from exc
