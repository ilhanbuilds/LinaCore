import pytest

from lina_core.core.exceptions import InferenceError
from lina_core.inference.generation import validate_stream
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.inference.response import InferenceResponse, TokenUsage
from lina_core.inference.sampling import GenerationOptions
from lina_core.inference.streaming import StreamEvent, StreamEventType


@pytest.mark.parametrize("role", list(ChatRole))
def test_roles(role: ChatRole):
    assert ChatRole(role.value) is role


def test_message():
    assert ChatMessage(ChatRole.USER, "hello").content == "hello"


@pytest.mark.parametrize("content", ["", " ", "\n"])
def test_empty_message(content: str):
    with pytest.raises(InferenceError):
        ChatMessage(ChatRole.USER, content)


def test_unsafe_metadata():
    with pytest.raises(InferenceError):
        ChatMessage(ChatRole.USER, "ok", metadata={"x": "password=hunter2"})


def test_empty_request():
    with pytest.raises(InferenceError):
        InferenceRequest(())


def test_system_not_first():
    with pytest.raises(InferenceError):
        InferenceRequest(
            (
                ChatMessage(ChatRole.USER, "x"),
                ChatMessage(ChatRole.SYSTEM, "s"),
                ChatMessage(ChatRole.USER, "y"),
            )
        )


def test_request_must_end_user():
    with pytest.raises(InferenceError):
        InferenceRequest((ChatMessage(ChatRole.USER, "x"), ChatMessage(ChatRole.ASSISTANT, "y")))


def test_valid_conversation():
    assert (
        len(
            InferenceRequest(
                (ChatMessage(ChatRole.SYSTEM, "s"), ChatMessage(ChatRole.USER, "x"))
            ).messages
        )
        == 2
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": 4097},
        {"temperature": -1},
        {"temperature": 3},
        {"top_p": 0},
        {"top_p": 2},
        {"top_k": -1},
        {"repetition_penalty": 0},
    ],
)
def test_invalid_generation_options(kwargs: dict[str, int]):
    with pytest.raises(InferenceError):
        GenerationOptions(**kwargs)  # type: ignore[arg-type]


def test_usage_total():
    assert TokenUsage(2, 3).total_tokens == 5


def test_usage_negative():
    with pytest.raises(ValueError):
        TokenUsage(-1, 2)


def test_response_latency():
    with pytest.raises(ValueError):
        InferenceResponse("r", "m", "x", "stop", TokenUsage(1, 1), -1)


def test_stream_valid():
    validate_stream(
        [
            StreamEvent(StreamEventType.STARTED, "r", index=0),
            StreamEvent(StreamEventType.COMPLETED, "r", index=1),
        ]
    )


def test_stream_invalid_start():
    with pytest.raises(ValueError):
        validate_stream([StreamEvent(StreamEventType.COMPLETED, "r", index=0)])


def test_stream_invalid_order():
    with pytest.raises(ValueError):
        validate_stream(
            [
                StreamEvent(StreamEventType.STARTED, "r", index=1),
                StreamEvent(StreamEventType.COMPLETED, "r", index=2),
            ]
        )
