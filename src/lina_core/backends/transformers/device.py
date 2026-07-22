from typing import Any

from lina_core.backends.device import DeviceSelection, DTypePolicy


def torch_dtype(selection: DeviceSelection, torch_module: Any) -> Any:
    mapping = {
        DTypePolicy.FLOAT32: torch_module.float32,
        DTypePolicy.FLOAT16: torch_module.float16,
        DTypePolicy.BFLOAT16: torch_module.bfloat16,
    }
    return mapping[selection.dtype]
