class LinaCoreError(Exception):
    """Base exception for safe, expected LinaCore failures."""


class ConfigurationError(LinaCoreError):
    pass


class RuntimeNotReadyError(LinaCoreError):
    pass


class ProviderNotFoundError(LinaCoreError):
    pass


class ModelLoadError(LinaCoreError):
    pass


class TokenizerError(LinaCoreError):
    pass


class InferenceError(LinaCoreError):
    pass


class DatasetError(LinaCoreError):
    pass


class TrainingError(LinaCoreError):
    pass


class EvaluationError(LinaCoreError):
    pass


class SecurityError(LinaCoreError):
    pass
