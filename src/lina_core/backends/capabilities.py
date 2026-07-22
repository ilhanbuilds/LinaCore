from enum import StrEnum


class BackendCapability(StrEnum):
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    STREAMING = "streaming"
    CANCELLATION = "cancellation"
    QUANTIZATION_INT8 = "quantization_int8"
    QUANTIZATION_INT4 = "quantization_int4"


class ModelFormat(StrEnum):
    TRANSFORMERS = "transformers"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    ONNX = "onnx"
