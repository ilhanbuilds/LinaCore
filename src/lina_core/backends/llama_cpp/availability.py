import importlib.util
from importlib.metadata import PackageNotFoundError, version

from lina_core.backends.base import BackendAvailability
from lina_core.diagnostics.hardware import detect_devices


def llama_cpp_availability() -> BackendAvailability:
    package = "llama-cpp-python"
    module_found = importlib.util.find_spec("llama_cpp") is not None
    detected: str | None = None
    if module_found:
        try:
            detected = version(package)
        except PackageNotFoundError:
            detected = "unknown"
    reason = (
        "llama.cpp dependency kullanılabilir; GGUF inference entegrasyonu foundation aşamasında."
        if module_found
        else "llama.cpp backend kullanılamıyor. Optional dependency grubunu yükleyin: "
        ".[inference-llama-cpp]"
    )
    warnings = ("Gerçek GGUF inference bu sprintte desteklenmiyor; yalnız foundation hazır.",)
    return BackendAvailability(
        module_found,
        "llama.cpp",
        {package: detected} if detected else {},
        () if module_found else (package,),
        detected,
        reason,
        detect_devices(),
        warnings,
    )
