import json

from service.prompting import build_prompt
from service.schemas import ChatMessage, ToolDefinition


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "tools": tools,
            }
        )
        return (
            json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False, sort_keys=True)
            + "<｜Assistant｜><think>\n"
        )


def test_build_prompt_uses_only_latest_user_message_for_plain_chat():
    tokenizer = FakeTokenizer()
    prompt = build_prompt(
        [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="Ping"),
        ],
        tokenizer=tokenizer,
    )

    call = tokenizer.calls[0]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert call["tools"] is None
    assert call["messages"] == [{"role": "user", "content": "Ping"}]
    assert '"content": "Ping"' in prompt
    assert "Hello" not in prompt
    assert "You are helpful." not in prompt


def test_build_prompt_injects_tool_instructions_for_first_tool_round():
    tokenizer = FakeTokenizer()
    prompt = build_prompt(
        [ChatMessage(role="user", content="What is the weather in Hangzhou?")],
        tokenizer=tokenizer,
        tools=[
            ToolDefinition(
                type="function",
                function={
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            )
        ],
        tool_choice="auto",
    )

    call = tokenizer.calls[0]
    assert call["tools"] is None
    assert call["messages"][0]["role"] == "user"
    assert "Available functions:" in call["messages"][0]["content"]
    assert '"name": "get_weather"' in call["messages"][0]["content"]
    assert "User: What is the weather in Hangzhou?" in call["messages"][0]["content"]
    assert '"name": "get_weather"' in prompt
    assert prompt.endswith("<｜Assistant｜>")
    assert not prompt.endswith("<｜Assistant｜><think>\n")


def test_build_prompt_keeps_tool_closure_after_latest_user():
    tokenizer = FakeTokenizer()
    prompt = build_prompt(
        [
            ChatMessage(role="user", content="old question"),
            ChatMessage(role="assistant", content="old answer"),
            ChatMessage(role="user", content="南京天气如何"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_whether",
                            "arguments": '{"city":"南京"}',
                        },
                    }
                ],
            ),
            ChatMessage(
                role="tool",
                content='{"city":"南京","weather":"sunny","temperature_c":25}',
                tool_call_id="call_1",
            ),
        ],
        tokenizer=tokenizer,
        tools=[
            ToolDefinition(
                type="function",
                function={
                    "name": "get_whether",
                    "description": "Get weather by city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            )
        ],
    )

    call = tokenizer.calls[0]
    assert call["messages"] == [
        {
            "role": "user",
            "content": (
                "You have already received the tool result for this user request.\n"
                "Use the tool result as the authoritative factual input.\n"
                "Do not explain the JSON structure.\n"
                "Do not describe the tool call process.\n"
                "Do not restate field names such as city, weather, temperature_c, status, message, or final.\n"
                "Answer the user's request directly in natural language.\n"
                "Keep the answer concise and user-facing.\n"
                "User: 南京天气如何"
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_whether",
                        "arguments": '{"city":"南京"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"city":"南京","weather":"sunny","temperature_c":25}',
            "tool_call_id": "call_1",
        },
    ]
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_whether",
                "description": "Get weather by city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    assert "old question" not in prompt
    assert prompt.endswith("<｜Assistant｜>")
    assert not prompt.endswith("<｜Assistant｜><think>\n")
    assert "Do not explain the JSON structure." in prompt
