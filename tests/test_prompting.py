from service.prompting import build_prompt
from service.schemas import ChatMessage


def test_build_prompt_orders_messages_and_wraps_template():
    prompt = build_prompt(
        [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ]
    )

    assert prompt.startswith("<|begin_of_sentence|><|User|>")
    assert "System: You are helpful." in prompt
    assert "User: Hello" in prompt
    assert "Assistant: Hi there" in prompt
    assert prompt.endswith("<|Assistant|>")
