import json

from fastapi.testclient import TestClient

from service.app import create_app
from service.rkllm_bridge import EngineConfig


class FakeEngine:
    def __init__(self) -> None:
        self.config = EngineConfig(
            model_path="fake.rkllm",
            bridge_lib_path="fake.so",
            tokenizer_path="fake-tokenizer",
            model_name="rkllm-local",
        )

    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        assert '"content": "ping"' in prompt
        assert request_id
        for token in ["po", "ng"]:
            on_token(token)
        return "pong"

    def close(self) -> None:
        return None


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
        return json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False, sort_keys=True)


def _make_app(engine_factory):
    tokenizer = FakeTokenizer()
    app = create_app(engine_factory=engine_factory, tokenizer_factory=lambda _: tokenizer)
    return app, tokenizer


def test_chat_completions_non_streaming():
    app, tokenizer = _make_app(FakeEngine)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "rkllm-local",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"]
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "pong"
    assert response.headers["X-Request-Id"] == payload["request_id"]
    assert tokenizer.calls[0]["messages"] == [{"role": "user", "content": "ping"}]


def test_chat_completions_streaming():
    app, tokenizer = _make_app(FakeEngine)
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "rkllm-local",
                "stream": True,
                "messages": [{"role": "user", "content": "ping"}],
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["X-Request-Id"]
    assert "chat.completion.chunk" in body
    assert '"request_id":' in body
    assert '"content": "po"' in body
    assert '"content": "ng"' in body
    assert "[DONE]" in body
    assert tokenizer.calls[0]["messages"] == [{"role": "user", "content": "ping"}]


class RawOutputEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        assert '"content": "ping"' in prompt
        assert request_id
        content = "<think>internal</think>final"
        on_token(content)
        return content


def test_chat_completions_preserve_raw_model_output():
    app, _ = _make_app(RawOutputEngine)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "rkllm-local",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "<think>internal</think>final"
    assert payload["usage"]["completion_tokens"] == 1


class ToolCallEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        assert "Available functions:" in prompt
        assert "User: What is the weather in Hangzhou?" in prompt
        assert '"name": "get_weather"' in prompt
        content = json.dumps(
            {
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Hangzhou"},
                        },
                    }
                ]
            }
        )
        on_token(content)
        return content


class InvalidToolCallEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        assert "Available functions:" in prompt
        assert '"name": "get_weather"' in prompt
        content = '{"tool_calls":[{"name":"missing_args"}]}'
        on_token(content)
        return content


class MalformedWeatherToolCallEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        assert "Available functions:" in prompt
        assert '"name": "get_weather"' in prompt
        content = (
            "I should call the weather function.\n\n"
            '{"tool_calls":[{"type":"function","function":{"name":"get_weather","arguments":{"city":"Hangzhou"}}]}}'
        )
        on_token(content)
        return content


def _tool_request_payload(stream: bool = False, tool_choice=None):
    payload = {
        "model": "rkllm-local",
        "stream": stream,
        "messages": [{"role": "user", "content": "What is the weather in Hangzhou?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather by city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def _add_request_payload(prompt: str = "Please add 12 and 30.", tool_choice=None):
    payload = {
        "model": "rkllm-local",
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "type": "function",
                "function": {
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
            }
        ],
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def test_chat_completions_return_tool_calls():
    app, tokenizer = _make_app(ToolCallEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_tool_request_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] is None
    tool_call = payload["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert tool_call["function"]["arguments"] == '{"city": "Hangzhou"}'
    assert tokenizer.calls[0]["tools"] is None
    assert "Available functions:" in tokenizer.calls[0]["messages"][0]["content"]


def test_chat_completions_keep_tool_result_messages_on_follow_up_round():
    class ToolFollowUpEngine(FakeEngine):
        def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
            assert '"tool_call_id": "call_1"' in prompt
            assert '"content": "{\\"city\\":\\"南京\\",\\"weather\\":\\"sunny\\",\\"temperature_c\\":25}"' in prompt
            assert "Available functions:" not in prompt
            content = "南京当前默认天气为 sunny，温度约25°C。"
            on_token(content)
            return content

    app, tokenizer = _make_app(ToolFollowUpEngine)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "rkllm-local",
                "messages": [
                    {"role": "user", "content": "南京天气如何"},
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
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_whether",
                            "description": "Get weather by city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert tokenizer.calls[0]["messages"] == [
        {"role": "user", "content": "南京天气如何"},
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
    assert tokenizer.calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_whether",
                "description": "Get weather by city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def test_chat_completions_accept_legacy_tool_call_shape():
    class LegacyToolCallEngine(FakeEngine):
        def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
            content = json.dumps(
                {
                    "tool_calls": [
                        {
                            "name": "get_weather",
                            "arguments": {"city": "Hangzhou"},
                        }
                    ]
                }
            )
            on_token(content)
            return content

    app, _ = _make_app(LegacyToolCallEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_tool_request_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = payload["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"


def test_chat_completions_repair_malformed_weather_tool_call():
    app, _ = _make_app(MalformedWeatherToolCallEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_tool_request_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = payload["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert tool_call["function"]["arguments"] == '{"city": "Hangzhou"}'


def test_chat_completions_degrade_on_invalid_tool_call_payload():
    app, _ = _make_app(InvalidToolCallEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_tool_request_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["message"]["content"] == '{"tool_calls":[{"name":"missing_args"}]}'


def test_chat_completions_reject_streaming_tool_calls():
    app, _ = _make_app(ToolCallEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_tool_request_payload(stream=True))

    assert response.status_code == 400
    assert "Streaming is not supported for tool calls yet" in response.json()["error"]["message"]


def test_chat_completions_accept_forced_tool_choice():
    app, _ = _make_app(ToolCallEngine)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_tool_request_payload(tool_choice={"type": "function", "function": {"name": "get_weather"}}),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


class AddStandardToolCallEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        assert "Available functions:" in prompt
        assert '"name": "add"' in prompt
        content = json.dumps(
            {
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": {"a": 12, "b": 30},
                        },
                    }
                ]
            }
        )
        on_token(content)
        return content


class AddLegacyToolCallEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        content = json.dumps({"tool_calls": [{"name": "add", "arguments": {"a": 12, "b": 30}}]})
        on_token(content)
        return content


class AddBrokenBraceEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        content = '{"tool_calls":[{"name":"add","arguments":{"a":12,"b":30}}]'
        on_token(content)
        return content


class AddTupleArgumentsEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        content = '{"tool_calls":[{"name":"add","arguments":"{12,30}"}]}'
        on_token(content)
        return content


class AddListArgumentsEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        content = json.dumps({"tool_calls": [{"name": "add", "arguments": [12, 30]}]})
        on_token(content)
        return content


class AddExpressionArgumentsEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        content = json.dumps({"tool_calls": [{"name": "add", "arguments": "12 + 30"}]})
        on_token(content)
        return content


class AddNamedTextArgumentsEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        content = json.dumps({"tool_calls": [{"name": "add", "arguments": "a=12, b=30"}]})
        on_token(content)
        return content


class AddSingleNumberEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        content = json.dumps({"tool_calls": [{"name": "add", "arguments": "12"}]})
        on_token(content)
        return content


class AddTooManyNumbersEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        content = json.dumps({"tool_calls": [{"name": "add", "arguments": "12, 30, 40"}]})
        on_token(content)
        return content


def _assert_add_tool_call_response(response):
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert payload["choices"][0]["message"]["content"] is None
    tool_call = payload["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "add"
    assert tool_call["function"]["arguments"] == '{"a": 12, "b": 30}'


def test_chat_completions_accept_add_standard_tool_call():
    app, _ = _make_app(AddStandardToolCallEngine)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_add_request_payload(tool_choice={"type": "function", "function": {"name": "add"}}),
        )

    _assert_add_tool_call_response(response)


def test_chat_completions_accept_add_legacy_shape():
    app, _ = _make_app(AddLegacyToolCallEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_add_request_payload())

    _assert_add_tool_call_response(response)


def test_chat_completions_repair_add_missing_closing_brace():
    app, _ = _make_app(AddBrokenBraceEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_add_request_payload())

    _assert_add_tool_call_response(response)


def test_chat_completions_repair_add_tuple_like_arguments():
    app, _ = _make_app(AddTupleArgumentsEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_add_request_payload())

    _assert_add_tool_call_response(response)


def test_chat_completions_repair_add_list_arguments():
    app, _ = _make_app(AddListArgumentsEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_add_request_payload())

    _assert_add_tool_call_response(response)


def test_chat_completions_repair_add_expression_arguments():
    app, _ = _make_app(AddExpressionArgumentsEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_add_request_payload())

    _assert_add_tool_call_response(response)


def test_chat_completions_repair_add_named_text_arguments():
    app, _ = _make_app(AddNamedTextArgumentsEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_add_request_payload())

    _assert_add_tool_call_response(response)


def test_chat_completions_do_not_repair_add_with_single_number():
    app, _ = _make_app(AddSingleNumberEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_add_request_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_do_not_repair_add_with_too_many_numbers():
    app, _ = _make_app(AddTooManyNumbersEngine)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_add_request_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "stop"
