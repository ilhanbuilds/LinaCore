import random
from typing import Any

from lina_core.backends.errors import GenerationError
from lina_core.inference.sampling import GenerationOptions
from lina_core.runtime.cancellation import CancellationToken


class CancellationStoppingCriteria:
    """Transformers-compatible callable that stops generation after cancellation."""

    def __init__(self, token: CancellationToken) -> None:
        self.token = token

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        del input_ids, scores, kwargs
        return self.token.cancelled


def generation_kwargs(options: GenerationOptions) -> dict[str, Any]:
    sampling = options.temperature > 0
    if not sampling and (options.top_p != 0.95 or options.top_k != 40):
        raise GenerationError("temperature=0 iken top_p/top_k sampling seçenekleri değiştirilemez.")
    values: dict[str, Any] = {
        "max_new_tokens": options.max_new_tokens,
        "do_sample": sampling,
        "repetition_penalty": options.repetition_penalty,
    }
    if sampling:
        values.update(temperature=options.temperature, top_p=options.top_p, top_k=options.top_k)
    return values


def apply_seed(seed: int | None, torch_module: Any) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
