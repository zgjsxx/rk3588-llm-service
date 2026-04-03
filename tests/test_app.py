from fastapi.testclient import TestClient

from service.app import create_app
from service.rkllm_bridge import EngineConfig


class FakeEngine:
    def __init__(self) -> None:
        self.config = EngineConfig(
            model_path="fake.rkllm",
            bridge_lib_path="fake.so",
            model_name="rkllm-local",
        )

    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        assert "User: ping" in prompt
        assert request_id
        for token in ["po", "ng"]:
            on_token(token)
        return "pong"

    def close(self) -> None:
        return None


def test_chat_completions_non_streaming():
    app = create_app(engine_factory=FakeEngine)
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


def test_chat_completions_streaming():
    app = create_app(engine_factory=FakeEngine)
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


class RawOutputEngine(FakeEngine):
    def generate(self, prompt: str, on_token, request_id: str, **_) -> str:
        assert "User: ping" in prompt
        assert request_id
        content = "<think>internal</think>final"
        on_token(content)
        return content


def test_chat_completions_preserve_raw_model_output():
    app = create_app(engine_factory=RawOutputEngine)
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
