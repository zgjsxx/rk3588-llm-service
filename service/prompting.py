from __future__ import annotations

import json
from typing import Any, Iterable

from .schemas import ChatMessage, ToolChoiceObject, ToolDefinition

DEFAULT_PROMPT_PREFIX = "<|begin_of_sentence|><|User|>"
DEFAULT_PROMPT_POSTFIX = "<|Assistant|>"


def _message_role_and_content(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role", "user"))
        content = str(message.get("content", "") or "")
        return role, content

    role = str(message.role)
    content = str(message.content or "")
    return role, content


def latest_user_content(messages: Iterable[ChatMessage | dict[str, str]]) -> str:
    latest = ""
    for message in messages:
        role, content = _message_role_and_content(message)
        if role == "user":
            latest = content.strip()
    return latest


def _format_tool_choice(tool_choice: str | ToolChoiceObject | dict[str, Any] | None) -> str:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        return f"required function: {tool_choice['function']['name']}"
    return f"required function: {tool_choice.function.name}"


def _tool_prompt_block(
    tools: list[ToolDefinition],
    tool_choice: str | ToolChoiceObject | dict[str, Any] | None,
) -> str:
    forced_tool_name = None
    if isinstance(tool_choice, dict):
        forced_tool_name = tool_choice["function"]["name"]
    elif isinstance(tool_choice, ToolChoiceObject):
        forced_tool_name = tool_choice.function.name

    formatted_tools = [
        json.dumps(
            {
                "name": tool.function.name,
                "description": tool.function.description or "",
                "parameters": tool.function.parameters,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for tool in tools
    ]
    sections = [
        "You are producing output for a tool-calling parser.",
        f"Tool choice policy: {_format_tool_choice(tool_choice)}.",
        "Available functions:",
        *formatted_tools,
        'If a function call is required, respond with JSON only in this exact OpenAI-compatible shape: {"tool_calls":[{"type":"function","function":{"name":"function_name","arguments":{"arg_name":"value"}}}]}',
        "The arguments field must be a JSON object whose keys match the function parameter names.",
        "Do not use arrays, sets, or positional arguments for arguments.",
        "Return raw JSON only.",
        "Do not output any explanation, analysis, prefix, suffix, markdown, or prose.",
        "Do not output <think> or chain-of-thought.",
        "Do not wrap the JSON in code fences.",
        "Make sure every opening brace and bracket is closed.",
        "Make sure the JSON is valid and parseable by Python json.loads().",
        'Valid example: {"tool_calls":[{"type":"function","function":{"name":"add","arguments":{"a":12,"b":30}}}]}',
        "Do not include markdown fences when returning JSON.",
        "If no function is needed, reply with a normal natural-language answer.",
    ]
    add_tool_names = {tool.function.name for tool in tools if tool.function.name == "add"}
    if add_tool_names and (len(tools) == 1 or forced_tool_name == "add"):
        sections.extend(
            [
                "For the add function, arguments must be exactly a JSON object with numeric keys a and b.",
                'Valid add example: {"tool_calls":[{"type":"function","function":{"name":"add","arguments":{"a":12,"b":30}}}]}',
                'Invalid add shape: {"tool_calls":[{"name":"add","arguments":{12,30}}]} because keys are missing.',
                'Invalid add shape: {"tool_calls":[{"name":"add","arguments":[12,30]}]} because arguments must be an object.',
                'Invalid add shape: {"tool_calls":[{"name":"add","arguments":"12,30"}]} because arguments must not be a string here.',
                "For add, do not answer with the final sum.",
            ]
        )
    if forced_tool_name is not None:
        sections.extend(
            [
                f"You must call the function named {forced_tool_name}.",
                "Do not answer the math yourself.",
                "Do not return natural language.",
                "Your entire response must be a single JSON object with tool_calls.",
            ]
        )
    return "\n".join(sections)


def build_prompt(
    messages: Iterable[ChatMessage | dict[str, str]],
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    prompt_postfix: str = DEFAULT_PROMPT_POSTFIX,
    tools: list[ToolDefinition] | None = None,
    tool_choice: str | ToolChoiceObject | dict[str, Any] | None = None,
) -> str:
    user_content = latest_user_content(messages)
    if not tools:
        return f"{prompt_prefix}User: {user_content}{prompt_postfix}"

    prompt_body = "\n\n".join(
        [
            _tool_prompt_block(tools, tool_choice),
            f"User: {user_content}",
        ]
    )
    return f"{prompt_prefix}{prompt_body}{prompt_postfix}"
