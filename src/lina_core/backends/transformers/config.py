from dataclasses import dataclass

from lina_core.backends.device import DTypePolicy, QuantizationPolicy
from lina_core.backends.errors import BackendError


@dataclass(frozen=True, slots=True)
class TransformersBackendConfig:
    model_path: str
    tokenizer_path: str | None = None
    model_id: str = "transformers-local"
    display_name: str = "Local Transformers Model"
    revision: str | None = None
    device: str = "auto"
    dtype: DTypePolicy = DTypePolicy.AUTO
    quantization: QuantizationPolicy = QuantizationPolicy.NONE
    local_files_only: bool = True
    allow_remote_downloads: bool = False
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    streaming: bool = True
    chat_template_policy: str = "required"

    def __post_init__(self) -> None:
        if not self.model_path:
            raise BackendError("Model yolu boş olamaz.")
        if self.trust_remote_code:
            raise BackendError("trust_remote_code güvenlik politikası gereği desteklenmiyor.")
        if not self.local_files_only and not self.allow_remote_downloads:
            raise BackendError("Uzak indirme için açık izin gereklidir.")
        if self.revision is not None and not self.revision.strip():
            raise BackendError("Revision boş olamaz.")
        if self.chat_template_policy not in {"required", "fallback"}:
            raise BackendError("Geçersiz chat template politikası.")
