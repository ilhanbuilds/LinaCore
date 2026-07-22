from dataclasses import dataclass

from lina_core.backends.errors import ContextLimitError
from lina_core.inference.chat_template import ChatTemplateRenderer, RenderedPrompt
from lina_core.inference.request import ChatMessage, ChatRole
from lina_core.tokenizer.base import TokenizerProvider


@dataclass(frozen=True, slots=True)
class TokenBudget:
    context_length: int
    prompt_tokens: int
    requested_output_tokens: int
    reserved_tokens: int
    available_tokens: int
    truncated: bool = False
    removed_messages: int = 0
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    messages: tuple[ChatMessage, ...]
    rendered: RenderedPrompt
    budget: TokenBudget


def prepare_prompt(
    messages: tuple[ChatMessage, ...],
    tokenizer: TokenizerProvider,
    renderer: ChatTemplateRenderer,
    *,
    context_length: int,
    requested_output_tokens: int,
    reserved_tokens: int = 0,
    truncation: str = "error",
) -> PreparedPrompt:
    current = messages
    removed = 0
    while True:
        rendered = renderer.render(current)
        prompt_tokens = tokenizer.count_tokens(rendered.text)
        available = context_length - prompt_tokens - reserved_tokens
        if requested_output_tokens <= available:
            budget = TokenBudget(
                context_length,
                prompt_tokens,
                requested_output_tokens,
                reserved_tokens,
                max(0, available),
                bool(removed),
                removed,
                "En eski konuşma mesajları kaldırıldı." if removed else None,
            )
            return PreparedPrompt(current, rendered, budget)
        if truncation != "oldest":
            raise ContextLimitError(
                f"İstek model context sınırını aşıyor: {prompt_tokens} + "
                f"{requested_output_tokens} > {context_length}."
            )
        removable = _oldest_removable_range(current)
        if removable is None:
            raise ContextLimitError(
                "System ve son user mesajı korunarak context sınırı sağlanamıyor."
            )
        start, end = removable
        removed += end - start
        current = current[:start] + current[end:]


def _oldest_removable_range(
    messages: tuple[ChatMessage, ...],
) -> tuple[int, int] | None:
    final_user = max(
        (index for index, message in enumerate(messages) if message.role is ChatRole.USER),
        default=-1,
    )
    for index, message in enumerate(messages):
        if message.role is ChatRole.USER and index != final_user:
            end = index + 1
            while end < final_user and messages[end].role is not ChatRole.USER:
                end += 1
            return index, end
    return None
