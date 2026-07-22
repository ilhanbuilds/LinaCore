from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransformersRuntimeMetadata:
    device: str
    dtype: str
    chat_template_fallback: bool
    local_files_only: bool
    model_path: str
