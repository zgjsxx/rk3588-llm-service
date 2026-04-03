import pytest
from pydantic import ValidationError

from service.schemas import ChatCompletionRequest, ChatMessage


def test_chat_completion_request_accepts_tools_and_tool_choice():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "rkllm-local",
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_status",
                        "description": "Look up service status",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup_status"}},
        }
    )

    assert request.tools is not None
    assert request.tools[0].function.name == "lookup_status"


def test_chat_completion_request_rejects_tool_choice_without_tools():
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {
                "model": "rkllm-local",
                "messages": [{"role": "user", "content": "ping"}],
                "tool_choice": "auto",
            }
        )


def test_tool_message_requires_tool_call_id():
    with pytest.raises(ValidationError):
        ChatMessage.model_validate({"role": "tool", "content": "{}"})


def test_assistant_message_accepts_tool_calls_with_empty_content():
    message = ChatMessage.model_validate(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_status",
                        "arguments": "{}",
                    },
                }
            ],
        }
    )

    assert message.tool_calls is not None
