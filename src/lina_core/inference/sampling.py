from dataclasses import dataclass

from lina_core.core.exceptions import InferenceError


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    seed: int | None = None
    stop: tuple[str, ...] = ()
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_new_tokens <= 4096:
            raise InferenceError("max_new_tokens must be between 1 and 4096")
        if not 0 <= self.temperature <= 2:
            raise InferenceError("temperature must be between 0 and 2")
        if not 0 < self.top_p <= 1:
            raise InferenceError("top_p must be between 0 and 1")
        if self.top_k < 0 or self.repetition_penalty <= 0:
            raise InferenceError("Invalid sampling value")
        if len(self.stop) > 16 or any(not item or len(item) > 100 for item in self.stop):
            raise InferenceError("Invalid stop sequences")
