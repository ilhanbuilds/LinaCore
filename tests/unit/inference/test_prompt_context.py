from typing import Any

import pytest

from lina_core.backends.errors import ContextLimitError
from lina_core.core.exceptions import TokenizerError
from lina_core.inference.chat_template import ChatTemplateRenderer
from lina_core.inference.prompt import prepare_prompt
from lina_core.inference.request import ChatMessage, ChatRole
from lina_core.tokenizer.mock import MockTokenizerProvider


class TemplateTokenizer:
    chat_template = "configured"

    def apply_chat_template(
        self,
        content: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        suffix = "assistant:" if add_generation_prompt else ""
        return "|".join(f"{item['role']}:{item['content']}" for item in content) + suffix


@pytest.mark.parametrize("role", list(ChatRole))
def test_fallback_renders_every_role(role: ChatRole) -> None:
    renderer = ChatTemplateRenderer(fallback_enabled=True)
    result = renderer.render((ChatMessage(role, "content"),))
    assert f"<{role.value}>" in result.text
    assert result.used_fallback and not result.used_chat_template


def test_fallback_adds_generation_prompt() -> None:
    result = ChatTemplateRenderer(fallback_enabled=True).render(
        (ChatMessage(ChatRole.USER, "hello"),)
    )
    assert result.text.endswith("<assistant>\n")


def test_fallback_can_omit_generation_prompt() -> None:
    result = ChatTemplateRenderer(fallback_enabled=True).render(
        (ChatMessage(ChatRole.USER, "hello"),), add_generation_prompt=False
    )
    assert not result.text.endswith("<assistant>\n")


def test_missing_template_requires_explicit_fallback() -> None:
    with pytest.raises(TokenizerError):
        ChatTemplateRenderer().render((ChatMessage(ChatRole.USER, "hello"),))


def test_native_template_is_preferred() -> None:
    result = ChatTemplateRenderer(TemplateTokenizer()).render(
        (ChatMessage(ChatRole.USER, "hello"),)
    )
    assert result.used_chat_template and not result.used_fallback
    assert result.text == "user:helloassistant:"


@pytest.mark.parametrize(
    "messages",
    [
        (ChatMessage(ChatRole.USER, "hello"),),
        (ChatMessage(ChatRole.SYSTEM, "rules"), ChatMessage(ChatRole.USER, "hello")),
        (
            ChatMessage(ChatRole.USER, "one"),
            ChatMessage(ChatRole.ASSISTANT, "two"),
            ChatMessage(ChatRole.USER, "three"),
        ),
    ],
)
def test_context_exact_limit(messages: tuple[ChatMessage, ...]) -> None:
    tokenizer = MockTokenizerProvider()
    renderer = ChatTemplateRenderer(fallback_enabled=True)
    rendered = renderer.render(messages)
    prompt_tokens = tokenizer.count_tokens(rendered.text)
    result = prepare_prompt(
        messages,
        tokenizer,
        renderer,
        context_length=prompt_tokens + 8,
        requested_output_tokens=8,
    )
    assert result.budget.available_tokens == 8
    assert not result.budget.truncated


@pytest.mark.parametrize("requested", [1, 8, 32, 128])
def test_context_overflow_is_controlled(requested: int) -> None:
    messages = (ChatMessage(ChatRole.USER, "long input"),)
    with pytest.raises(ContextLimitError):
        prepare_prompt(
            messages,
            MockTokenizerProvider(),
            ChatTemplateRenderer(fallback_enabled=True),
            context_length=requested,
            requested_output_tokens=requested,
        )


def test_truncation_removes_oldest_complete_turn() -> None:
    messages = (
        ChatMessage(ChatRole.SYSTEM, "keep system"),
        ChatMessage(ChatRole.USER, "old user " * 20),
        ChatMessage(ChatRole.ASSISTANT, "old assistant " * 20),
        ChatMessage(ChatRole.USER, "keep final"),
    )
    result = prepare_prompt(
        messages,
        MockTokenizerProvider(),
        ChatTemplateRenderer(fallback_enabled=True),
        context_length=100,
        requested_output_tokens=10,
        truncation="oldest",
    )
    assert result.budget.removed_messages == 2
    assert [item.role for item in result.messages] == [ChatRole.SYSTEM, ChatRole.USER]
    assert result.messages[-1].content == "keep final"


def test_truncation_preserves_system_message() -> None:
    messages = (
        ChatMessage(ChatRole.SYSTEM, "system"),
        ChatMessage(ChatRole.USER, "old " * 30),
        ChatMessage(ChatRole.ASSISTANT, "answer " * 30),
        ChatMessage(ChatRole.USER, "final"),
    )
    result = prepare_prompt(
        messages,
        MockTokenizerProvider(),
        ChatTemplateRenderer(fallback_enabled=True),
        context_length=90,
        requested_output_tokens=5,
        truncation="oldest",
    )
    assert result.messages[0].role is ChatRole.SYSTEM


def test_truncation_cannot_remove_only_final_user() -> None:
    messages = (ChatMessage(ChatRole.USER, "too long " * 100),)
    with pytest.raises(ContextLimitError):
        prepare_prompt(
            messages,
            MockTokenizerProvider(),
            ChatTemplateRenderer(fallback_enabled=True),
            context_length=20,
            requested_output_tokens=5,
            truncation="oldest",
        )


def test_budget_reports_reserved_tokens() -> None:
    messages = (ChatMessage(ChatRole.USER, "x"),)
    result = prepare_prompt(
        messages,
        MockTokenizerProvider(),
        ChatTemplateRenderer(fallback_enabled=True),
        context_length=100,
        requested_output_tokens=5,
        reserved_tokens=7,
    )
    assert result.budget.reserved_tokens == 7


def test_template_tokenizer_arguments_are_plain_data() -> None:
    tokenizer: Any = TemplateTokenizer()
    rendered = ChatTemplateRenderer(tokenizer).render((ChatMessage(ChatRole.TOOL, "result"),))
    assert "tool:result" in rendered.text
