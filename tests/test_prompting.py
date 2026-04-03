from service.prompting import build_prompt
from service.schemas import ChatMessage, ToolDefinition


def test_build_prompt_uses_only_latest_user_message():
    prompt = build_prompt(
        [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="tool", content='{"temp": 20}', tool_call_id="call_1"),
            ChatMessage(role="user", content="Ping"),
        ]
    )

    assert prompt.startswith("<|begin_of_sentence|><|User|>")
    assert "User: Ping" in prompt
    assert "Hello" not in prompt
    assert "You are helpful." not in prompt
    assert "Hi there" not in prompt
    assert prompt.endswith("<|Assistant|>")


def test_build_prompt_includes_tooling_instructions():
    prompt = build_prompt(
        [ChatMessage(role="user", content="What is the weather in Hangzhou?")],
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

    assert "Available functions:" in prompt
    assert '"name": "get_weather"' in prompt
    assert '"required": ["city"]' in prompt
    assert "Tool choice policy: auto." in prompt
    assert 'respond with JSON only in this exact shape' in prompt
    assert "User: What is the weather in Hangzhou?" in prompt


def test_build_prompt_respects_forced_tool_choice():
    prompt = build_prompt(
        [ChatMessage(role="user", content="Ping")],
        tools=[
            ToolDefinition(
                type="function",
                function={"name": "lookup_status", "description": None, "parameters": {}},
            )
        ],
        tool_choice={"type": "function", "function": {"name": "lookup_status"}},
    )

    assert "Tool choice policy: required function: lookup_status." in prompt


def test_build_prompt_add_includes_strict_rules():
    prompt = build_prompt(
        [ChatMessage(role="user", content="Please add 12 and 30.")],
        tools=[
            ToolDefinition(
                type="function",
                function={
                    "name": "add",
                    "description": "Add two numbers and return their sum.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            )
        ],
        tool_choice={"type": "function", "function": {"name": "add"}},
    )

    assert 'Valid add example: {"tool_calls":[{"type":"function","function":{"name":"add","arguments":{"a":12,"b":30}}}]}' in prompt
    assert 'Invalid add shape: {"tool_calls":[{"name":"add","arguments":{12,30}}]}' in prompt
    assert "For add, do not answer with the final sum." in prompt
