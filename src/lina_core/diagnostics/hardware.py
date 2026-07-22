import importlib.util
import platform
from typing import Any

from lina_core.backends.device import DeviceInfo, DeviceType


def detect_devices(torch_module: Any | None = None) -> tuple[DeviceInfo, ...]:
    devices: list[DeviceInfo] = [
        DeviceInfo(DeviceType.CPU, display_name=platform.processor() or "CPU")
    ]
    torch = torch_module
    if torch is None and importlib.util.find_spec("torch") is not None:
        try:
            import torch as detected_torch

            torch = detected_torch
        except ImportError:
            torch = None
    if torch is not None and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                DeviceInfo(
                    DeviceType.CUDA,
                    index,
                    properties.name,
                    int(properties.total_memory),
                    compute_capability=f"{properties.major}.{properties.minor}",
                    recommended=index == 0,
                )
            )
    if (
        torch is not None
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        devices.append(DeviceInfo(DeviceType.MPS, display_name="Apple Metal", recommended=True))
    return tuple(devices)
