from __future__ import annotations

from typing import Any, Iterable

from .schemas import ChatMessage

DEFAULT_PROMPT_PREFIX = "<｜begin▁of▁sentence｜><｜User｜>"
DEFAULT_PROMPT_POSTFIX = "<｜Assistant｜>"


def _message_role_and_content(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        return role, content

    role = str(message.role)
    content = str(message.content)
    return role, content


def latest_user_content(messages: Iterable[ChatMessage | dict[str, str]]) -> str:
    latest = ""
    for message in messages:
        role, content = _message_role_and_content(message)
        if role == "user":
            latest = content.strip()
    return latest


def build_prompt(
    messages: Iterable[ChatMessage | dict[str, str]],
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    prompt_postfix: str = DEFAULT_PROMPT_POSTFIX,
) -> str:
    user_content = latest_user_content(messages)
    return f"{prompt_prefix}{user_content}{prompt_postfix}"
