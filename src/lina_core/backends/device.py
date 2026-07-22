from dataclasses import dataclass, field
from enum import StrEnum

from lina_core.backends.errors import DeviceSelectionError, DTypeCompatibilityError


class DeviceType(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    DIRECTML = "directml"


class DTypePolicy(StrEnum):
    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class QuantizationPolicy(StrEnum):
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    device_type: DeviceType
    index: int | None = None
    display_name: str = "CPU"
    total_memory: int | None = None
    available_memory: int | None = None
    compute_capability: str | None = None
    backend_compatibility: tuple[str, ...] = ("transformers",)
    recommended: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def identifier(self) -> str:
        return (
            f"{self.device_type}:{self.index}" if self.index is not None else self.device_type.value
        )


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    device: DeviceInfo
    dtype: DTypePolicy
    quantization: QuantizationPolicy = QuantizationPolicy.NONE
    warnings: tuple[str, ...] = field(default_factory=tuple)


def select_device(
    policy: str,
    devices: tuple[DeviceInfo, ...],
    dtype: DTypePolicy = DTypePolicy.AUTO,
    quantization: QuantizationPolicy = QuantizationPolicy.NONE,
) -> DeviceSelection:
    if quantization is not QuantizationPolicy.NONE:
        raise DTypeCompatibilityError(
            "Bu quantization seçeneği için gerekli optional bağımlılık kurulu değil."
        )
    requested = policy.casefold()
    if not devices:
        raise DeviceSelectionError("Kullanılabilir device bulunamadı.")
    device: DeviceInfo | None
    if requested == "auto":
        device = next((item for item in devices if item.recommended), devices[0])
    elif requested == "cpu":
        device = next((item for item in devices if item.device_type is DeviceType.CPU), None)
    elif requested == "mps":
        device = next((item for item in devices if item.device_type is DeviceType.MPS), None)
    elif requested == "cuda" or requested.startswith("cuda:"):
        index = 0 if requested == "cuda" else _parse_cuda_index(requested)
        device = next(
            (
                item
                for item in devices
                if item.device_type is DeviceType.CUDA and item.index == index
            ),
            None,
        )
    else:
        raise DeviceSelectionError(f"Desteklenmeyen device politikası: {policy}")
    if device is None:
        raise DeviceSelectionError(f"İstenen device kullanılamıyor: {policy}")
    actual_dtype = (
        DTypePolicy.FLOAT32
        if dtype is DTypePolicy.AUTO and device.device_type is DeviceType.CPU
        else dtype
    )
    if actual_dtype is DTypePolicy.AUTO:
        actual_dtype = DTypePolicy.FLOAT16
    if device.device_type is DeviceType.CPU and actual_dtype is DTypePolicy.FLOAT16:
        raise DTypeCompatibilityError("CPU üzerinde float16 sessizce kullanılamaz; float32 seçin.")
    return DeviceSelection(device, actual_dtype, quantization)


def _parse_cuda_index(value: str) -> int:
    try:
        index = int(value.partition(":")[2])
    except ValueError as exc:
        raise DeviceSelectionError(f"Geçersiz CUDA device: {value}") from exc
    if index < 0:
        raise DeviceSelectionError(f"Geçersiz CUDA device: {value}")
    return index
