from __future__ import annotations

import json
from typing import Any, Iterable, Protocol

from transformers import AutoTokenizer

from .schemas import ChatMessage, ToolDefinition


_ASSISTANT_THINK_SUFFIX = "<｜Assistant｜><think>\n"
_ASSISTANT_SUFFIX = "<｜Assistant｜>"
_TOOL_OUTPUTS_END_SUFFIX = "<｜tool▁outputs▁end｜>"


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str: ...


def load_tokenizer(tokenizer_path: str) -> ChatTemplateTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _message_role_and_content(message: Any) -> tuple[str, str | None]:
    if isinstance(message, dict):
        role = str(message.get("role", "user"))
        content = message.get("content")
        return role, None if content is None else str(content)

    role = str(message.role)
    content = message.content
    return role, None if content is None else str(content)


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


def _normalize_message(message: ChatMessage | dict[str, Any]) -> dict[str, Any]:
    role, content = _message_role_and_content(message)
    payload: dict[str, Any] = {"role": role, "content": content}

    tool_call_id = _message_tool_call_id(message)
    if tool_call_id is not None:
        payload["tool_call_id"] = tool_call_id

    tool_calls = _message_tool_calls(message)
    if tool_calls:
        payload["tool_calls"] = [_normalize_tool_call_payload(item) for item in tool_calls]

    return payload


def _normalize_tool(tool: ToolDefinition | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool, dict):
        return tool
    return tool.model_dump()


def _format_tool_choice(tool_choice: str | dict[str, Any] | None) -> str:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    return f"required function: {tool_choice['function']['name']}"


def _select_messages(messages: Iterable[ChatMessage | dict[str, Any]]) -> list[dict[str, Any]]:
    items = [_normalize_message(message) for message in messages]

    # The demo intentionally keeps prompt history tiny on-device: we only keep the
    # latest user turn, plus any assistant/tool closure generated for that turn.
    latest_user_index = -1
    for index, message in enumerate(items):
        if message["role"] == "user":
            latest_user_index = index

    if latest_user_index < 0:
        return []

    selected: list[dict[str, Any]] = [items[latest_user_index]]
    for message in items[latest_user_index + 1 :]:
        if message["role"] in {"assistant", "tool"}:
            selected.append(message)

    return selected


def _has_tool_closure(messages: list[dict[str, Any]]) -> bool:
    return any(message["role"] in {"assistant", "tool"} for message in messages[1:])


def _render_first_tool_call_user_text(
    user_content: str,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None,
) -> str:
    # DeepSeek's chat template does not expose available tools on the first round,
    # so we inject a compact tool catalogue into the current user message only for
    # the "discover a tool call" turn.
    sections = [
        "You are producing output for a tool-calling parser.",
        f"Tool choice policy: {_format_tool_choice(tool_choice)}.",
        "Available functions:",
        *[json.dumps(tool["function"], ensure_ascii=False, sort_keys=True) for tool in tools],
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
        f"User: {user_content}",
    ]

    add_tools = [tool for tool in tools if tool["function"]["name"] == "add"]
    if add_tools and (len(tools) == 1 or _format_tool_choice(tool_choice) == "required function: add"):
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

    if isinstance(tool_choice, dict):
        forced_tool_name = tool_choice["function"]["name"]
        sections.extend(
            [
                f"You must call the function named {forced_tool_name}.",
                "Do not answer the task yourself.",
                "Do not return natural language.",
                "Your entire response must be a single JSON object with tool_calls.",
            ]
        )

    return "\n".join(sections)


def _render_tool_follow_up_user_text(user_content: str) -> str:
    return "\n".join(
        [
            "You have already received the tool result for this user request.",
            "Use the tool result as the authoritative factual input.",
            "Do not explain the JSON structure.",
            "Do not describe the tool call process.",
            "Do not restate field names such as city, weather, temperature_c, status, message, or final.",
            "Answer the user's request directly in natural language.",
            "Keep the answer concise and user-facing.",
            f"User: {user_content}",
        ]
    )


def _strip_generation_think_suffix(prompt: str) -> str:
    if prompt.endswith(_ASSISTANT_THINK_SUFFIX):
        # For the first tool-call round we want JSON immediately, not a reasoning
        # prelude that tends to push the small model into `<think>` output.
        return prompt[: -len(_ASSISTANT_THINK_SUFFIX)] + _ASSISTANT_SUFFIX
    return prompt


def _ensure_assistant_generation_suffix(prompt: str, *, allow_think: bool) -> str:
    if prompt.endswith(_ASSISTANT_THINK_SUFFIX) or prompt.endswith(_ASSISTANT_SUFFIX):
        return prompt
    if prompt.endswith(_TOOL_OUTPUTS_END_SUFFIX):
        # After tool outputs the model still needs an assistant-generation cue;
        # otherwise it may keep imitating tool protocol tokens instead of replying.
        return prompt + (_ASSISTANT_THINK_SUFFIX if allow_think else _ASSISTANT_SUFFIX)
    return prompt


def build_prompt(
    messages: Iterable[ChatMessage | dict[str, Any]],
    tokenizer: ChatTemplateTokenizer,
    tools: list[ToolDefinition | dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> str:
    selected_messages = _select_messages(messages)
    normalized_tools = None
    first_tool_round = False
    if tools and tool_choice != "none":
        normalized_tools = [_normalize_tool(tool) for tool in tools]

    if normalized_tools and selected_messages and not _has_tool_closure(selected_messages):
        first_tool_round = True
        selected_messages = [
            {
                "role": "user",
                "content": _render_first_tool_call_user_text(
                    selected_messages[0].get("content") or "",
                    normalized_tools,
                    tool_choice,
                ),
            }
        ]
        normalized_tools = None

    prompt = tokenizer.apply_chat_template(
        selected_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=normalized_tools,
    )
    if first_tool_round:
        prompt = _strip_generation_think_suffix(prompt)
    elif _has_tool_closure(selected_messages):
        # Follow-up rounds keep the minimal closure for the latest request:
        # latest user + assistant.tool_calls + tool result, then ask the model to
        # answer from the tool result instead of explaining the payload.
        selected_messages = [
            {
                **selected_messages[0],
                "content": _render_tool_follow_up_user_text(selected_messages[0].get("content") or ""),
            },
            *selected_messages[1:],
        ]
        prompt = tokenizer.apply_chat_template(
            selected_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=normalized_tools,
        )
        prompt = _ensure_assistant_generation_suffix(prompt, allow_think=False)
    return prompt
