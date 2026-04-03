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


def _message_tool_call_id(message: Any) -> str | None:
    if isinstance(message, dict):
        value = message.get("tool_call_id")
        return str(value) if value else None
    value = getattr(message, "tool_call_id", None)
    return str(value) if value else None


def _message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        value = message.get("tool_calls")
        return value if isinstance(value, list) else []
    value = getattr(message, "tool_calls", None)
    return list(value) if value else []


def _normalize_tool_call_payload(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return tool_call
    function = getattr(tool_call, "function", None)
    function_payload: dict[str, Any] | None = None
    if function is not None:
        function_payload = {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", ""),
        }
    return {
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "function"),
        "function": function_payload,
    }


def _tool_parts(tool: ToolDefinition | dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if isinstance(tool, dict):
        function = dict(tool.get("function", {}) or {})
        return (
            str(function.get("name", "")),
            str(function.get("description", "") or ""),
            dict(function.get("parameters", {}) or {}),
        )
    return (
        tool.function.name,
        tool.function.description or "",
        dict(tool.function.parameters or {}),
    )


def _messages_after_latest_user(messages: Iterable[ChatMessage | dict[str, Any]]) -> list[Any]:
    items = list(messages)
    latest_user_index = -1
    for index, message in enumerate(items):
        role, _ = _message_role_and_content(message)
        if role == "user":
            latest_user_index = index
    if latest_user_index < 0:
        return []
    return items[latest_user_index + 1 :]


def _tool_result_prompt_block(messages: Iterable[ChatMessage | dict[str, Any]]) -> str:
    rows: list[str] = []
    for message in _messages_after_latest_user(messages):
        role, content = _message_role_and_content(message)
        if role == "assistant":
            tool_calls = _message_tool_calls(message)
            if tool_calls:
                normalized = [_normalize_tool_call_payload(item) for item in tool_calls]
                rows.append(
                    f"Assistant tool calls: {json.dumps(normalized, ensure_ascii=False, sort_keys=True)}"
                )
        elif role == "tool":
            tool_call_id = _message_tool_call_id(message) or ""
            rows.append(f"Tool result ({tool_call_id}): {content}")
    return "\n".join(rows)


def _format_tool_choice(tool_choice: str | ToolChoiceObject | dict[str, Any] | None) -> str:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        return f"required function: {tool_choice['function']['name']}"
    return f"required function: {tool_choice.function.name}"


def _tool_prompt_block(
    tools: list[ToolDefinition | dict[str, Any]],
    tool_choice: str | ToolChoiceObject | dict[str, Any] | None,
) -> str:
    forced_tool_name = None
    if isinstance(tool_choice, dict):
        forced_tool_name = tool_choice["function"]["name"]
    elif isinstance(tool_choice, ToolChoiceObject):
        forced_tool_name = tool_choice.function.name

    formatted_tools = []
    for tool in tools:
        name, description, parameters = _tool_parts(tool)
        formatted_tools.append(
            json.dumps(
                {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
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
    add_tool_names = {name for name, _, _ in (_tool_parts(tool) for tool in tools) if name == "add"}
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
    tools: list[ToolDefinition | dict[str, Any]] | None = None,
    tool_choice: str | ToolChoiceObject | dict[str, Any] | None = None,
) -> str:
    user_content = latest_user_content(messages)
    if not tools:
        return f"{prompt_prefix}User: {user_content}{prompt_postfix}"

    tool_result_block = _tool_result_prompt_block(messages)
    if tool_result_block:
        prompt_body = "\n\n".join(
            [
                "You are answering the user after tool execution.",
                "The tool results below are authoritative.",
                "Do not call any tool again if the tool results already answer the question.",
                "Do not output tool_calls JSON.",
                "Reply with a concise natural-language answer for the user.",
                f"User: {user_content}",
                tool_result_block,
            ]
        )
        return f"{prompt_prefix}{prompt_body}{prompt_postfix}"

    prompt_body = "\n\n".join(
        [
            _tool_prompt_block(tools, tool_choice),
            f"User: {user_content}",
        ]
    )
    return f"{prompt_prefix}{prompt_body}{prompt_postfix}"
