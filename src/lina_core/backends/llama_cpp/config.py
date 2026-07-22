from dataclasses import dataclass
from pathlib import Path

from lina_core.backends.errors import BackendError
from lina_core.security.paths import safe_resolve


@dataclass(frozen=True, slots=True)
class LlamaCppBackendConfig:
    model_path: Path
    model_root: Path
    model_id: str = "gguf-local"
    context_length: int = 2048
    threads: int | None = None
    gpu_layers: int = 0
    seed: int = 42
    verbose: bool = False

    def __post_init__(self) -> None:
        resolved = safe_resolve(self.model_root, self.model_path)
        if resolved.suffix.casefold() != ".gguf":
            raise BackendError("llama.cpp backend yalnız .gguf model dosyalarını kabul eder.")
        if not self.model_id:
            raise BackendError("Model kimliği boş olamaz.")
        if self.context_length <= 0:
            raise BackendError("Context length pozitif olmalıdır.")
        if self.threads is not None and self.threads <= 0:
            raise BackendError("Thread sayısı pozitif olmalıdır.")
        if self.gpu_layers < 0:
            raise BackendError("GPU layer sayısı negatif olamaz.")

    @property
    def resolved_model_path(self) -> Path:
        return safe_resolve(self.model_root, self.model_path)
