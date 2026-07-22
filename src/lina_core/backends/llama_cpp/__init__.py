"""Optional llama.cpp/GGUF backend foundation."""

from lina_core.backends.llama_cpp.backend import LlamaCppBackend
from lina_core.backends.llama_cpp.config import LlamaCppBackendConfig

__all__ = ["LlamaCppBackend", "LlamaCppBackendConfig"]
