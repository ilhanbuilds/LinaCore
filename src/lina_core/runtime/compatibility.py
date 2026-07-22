from dataclasses import dataclass

from lina_core.backends.device import DeviceSelection, DeviceType, DTypePolicy


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    compatible: bool
    warnings: tuple[str, ...] = ()
    reason: str | None = None


def assess_model_size(
    parameter_count: int | None, selection: DeviceSelection
) -> CompatibilityResult:
    if parameter_count is None:
        return CompatibilityResult(True)
    if selection.device.device_type is DeviceType.CPU and parameter_count > 7_000_000_000:
        return CompatibilityResult(
            True,
            ("Model CPU inference için çok büyük olabilir.",),
        )
    if selection.dtype is DTypePolicy.FLOAT16 and selection.device.device_type is DeviceType.CPU:
        return CompatibilityResult(False, reason="CPU ve float16 uyumlu değil.")
    return CompatibilityResult(True)
