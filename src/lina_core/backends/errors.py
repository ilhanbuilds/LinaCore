from lina_core.core.exceptions import LinaCoreError


class BackendError(LinaCoreError):
    """Base exception for controlled backend failures."""


class BackendUnavailableError(BackendError):
    """Raised when optional backend dependencies are unavailable."""


class DeviceSelectionError(BackendError):
    """Raised when a requested device cannot be used."""


class DTypeCompatibilityError(BackendError):
    """Raised for unsafe device and dtype combinations."""


class ContextLimitError(BackendError):
    """Raised when prompt and output exceed the model context."""


class GenerationError(BackendError):
    """Raised when generation fails without exposing backend internals."""
