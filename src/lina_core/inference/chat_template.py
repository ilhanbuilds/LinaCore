from dataclasses import dataclass
from typing import Any

from lina_core.core.exceptions import TokenizerError
from lina_core.inference.request import ChatMessage


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    used_chat_template: bool
    used_fallback: bool


class ChatTemplateRenderer:
    def __init__(self, tokenizer: Any | None = None, *, fallback_enabled: bool = False) -> None:
        self._tokenizer = tokenizer
        self._fallback_enabled = fallback_enabled

    def render(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        add_generation_prompt: bool = True,
    ) -> RenderedPrompt:
        if self._tokenizer is not None and getattr(self._tokenizer, "chat_template", None):
            content = [{"role": item.role.value, "content": item.content} for item in messages]
            text = self._tokenizer.apply_chat_template(
                content,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            return RenderedPrompt(str(text), True, False)
        if not self._fallback_enabled:
            raise TokenizerError(
                "Tokenizer chat template içermiyor; deterministic fallback "
                "açıkça etkinleştirilmedi."
            )
        lines = [
            f"<{message.role.value}>\n{message.content}\n</{message.role.value}>"
            for message in messages
        ]
        if add_generation_prompt:
            lines.append("<assistant>\n")
        return RenderedPrompt("\n".join(lines), False, True)
