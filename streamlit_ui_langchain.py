from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx
import streamlit as st
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from service.response_cleaning import split_response_text
from service.trace import trace_event


DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_MODEL = "rkllm-local"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant using tools when needed. "
    "Tool outputs are authoritative. "
    "Do not call the same tool more than once in a single turn with identical arguments. "
    "If a tool response contains duplicate_call_blocked=true, do not call that tool again with the same arguments. "
    "Use the tool result directly and answer the user."
)

WEATHER_DATA = {
    "beijing": {"city": "Beijing", "weather": "sunny", "temperature_c": 22},
    "shanghai": {"city": "Shanghai", "weather": "cloudy", "temperature_c": 24},
    "hangzhou": {"city": "Hangzhou", "weather": "rainy", "temperature_c": 20},
    "shenzhen": {"city": "Shenzhen", "weather": "humid", "temperature_c": 28},
}
DEFAULT_WEATHER = {"weather": "sunny", "temperature_c": 25}


def init_state() -> None:
    defaults = {
        "messages": [],
        "agent_messages": [],
        "api_base": DEFAULT_API_BASE,
        "model_name": DEFAULT_MODEL,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "temperature": 0.8,
        "max_tokens": 2048,
        "timeout": 120,
        "show_trace_details": False,
        "stream_responses": True,
        "last_trace": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add(a: float, b: float) -> str:
    """Add two numbers and return their sum."""
    payload = {
        "status": "ok",
        "operation": "add",
        "a": a,
        "b": b,
        "result": a + b,
        "final": True,
    }
    return json.dumps(payload, ensure_ascii=False)


def get_whether(city: str) -> str:
    """Get predefined weather data by city name."""
    city_name = city.strip()
    if not city_name:
        return json.dumps(
            {
                "status": "error",
                "tool": "get_whether",
                "message": "city is required",
                "final": True,
            },
            ensure_ascii=False,
        )

    payload = WEATHER_DATA.get(city_name.lower(), {"city": city_name, **DEFAULT_WEATHER})
    return json.dumps(
        {
            **payload,
            "status": "ok",
            "message": f"Weather lookup completed for {payload['city']}.",
            "final": True,
        },
        ensure_ascii=False,
    )


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            think_text, answer_text = split_response_text(message["content"])
            if think_text:
                with st.expander("Thought Process", expanded=False):
                    st.markdown(think_text)
            st.markdown(answer_text or message["content"])


def create_langchain_agent(request_id: str) -> Any:
    model = ChatOpenAI(
        model=st.session_state.model_name,
        api_key="dummy",
        base_url=f"{st.session_state.api_base.rstrip('/')}/v1",
        temperature=float(st.session_state.temperature),
        max_tokens=int(st.session_state.max_tokens),
        timeout=float(st.session_state.timeout),
        extra_body={"request_id": request_id},
    )
    return create_agent(
        model=model,
        tools=[add, get_whether],
        system_prompt=st.session_state.system_prompt or DEFAULT_SYSTEM_PROMPT,
    )


def create_chat_model(request_id: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=st.session_state.model_name,
        api_key="dummy",
        base_url=f"{st.session_state.api_base.rstrip('/')}/v1",
        temperature=float(st.session_state.temperature),
        max_tokens=int(st.session_state.max_tokens),
        timeout=float(st.session_state.timeout),
        extra_body={"request_id": request_id},
    )


def normalize_agent_output(result: Any) -> tuple[str, list[dict[str, Any]]]:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    final_text = ""
    tool_runs: list[dict[str, Any]] = []

    for message in messages:
        message_type = getattr(message, "type", None)
        if message_type == "tool":
            tool_runs.append(
                {
                    "name": getattr(message, "name", ""),
                    "tool_call_id": getattr(message, "tool_call_id", ""),
                    "content": getattr(message, "content", ""),
                }
            )
        elif message_type == "ai":
            content = getattr(message, "content", "")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
                final_text = "".join(parts).strip() or final_text
            elif isinstance(content, str):
                final_text = content.strip() or final_text

    if not final_text and messages:
        final_text = str(getattr(messages[-1], "content", "")).strip()
    return final_text, tool_runs


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return ""


def stream_plain_chat(model: ChatOpenAI, messages: list[dict[str, str]]) -> str:
    placeholder = st.empty()
    collected = ""
    for chunk in model.stream(messages):
        text = _chunk_text(chunk)
        if not text:
            continue
        collected += text
        think_text, answer_text = split_response_text(collected)
        placeholder.markdown(answer_text or collected)
    placeholder.empty()
    return collected.strip()


def healthcheck(base_url: str, timeout: float) -> None:
    response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
    response.raise_for_status()


def _extract_weather_city(prompt: str) -> str | None:
    text = prompt.strip()
    if not text:
        return None

    chinese_match = re.search(r"([\u4e00-\u9fffA-Za-z]+?)天气", text)
    if chinese_match:
        return chinese_match.group(1).strip("的是在请问一下呢吗呀，。？！ ") or None

    english_match = re.search(r"weather\s+(?:in|for)\s+([A-Za-z\s]+)", text, re.IGNORECASE)
    if english_match:
        return english_match.group(1).strip(" ?!.,")
    return None


def _extract_add_args(prompt: str) -> tuple[float, float] | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


def should_use_tooling(prompt: str) -> bool:
    text = prompt.strip().lower()
    if not text:
        return False
    if _extract_weather_city(prompt):
        return True
    if _extract_add_args(prompt):
        return True
    tool_keywords = ("tool", "function", "weather", "add", "sum", "天气", "加", "相加")
    return any(keyword in text for keyword in tool_keywords)


def fallback_tool_answer(prompt: str) -> str | None:
    city = _extract_weather_city(prompt)
    if city:
        payload = json.loads(get_whether(city))
        return f"{payload['city']}当前默认天气为{payload['weather']}，温度约{payload['temperature_c']}°C。"

    add_args = _extract_add_args(prompt)
    if add_args:
        payload = json.loads(add(*add_args))
        return f"{payload['a']} + {payload['b']} = {payload['result']}"

    return None


def emit_ui_agent_request_trace(
    request_id: str,
    *,
    messages: list[dict[str, Any]],
    mode: str,
) -> None:
    trace_event(
        "ui.agent.request",
        request_id=request_id,
        api_base=st.session_state.api_base,
        model=st.session_state.model_name,
        mode=mode,
        messages=messages,
    )


def emit_ui_agent_response_trace(
    request_id: str,
    *,
    mode: str,
    status: str,
    response_text: str,
    tool_runs: list[dict[str, Any]] | None = None,
    error: str | None = None,
    agent_result: Any = None,
    fallback_used: bool = False,
) -> None:
    trace_event(
        "ui.agent.response",
        request_id=request_id,
        api_base=st.session_state.api_base,
        model=st.session_state.model_name,
        mode=mode,
        status=status,
        response_text=response_text,
        tool_runs=tool_runs or [],
        error=error,
        agent_result=str(agent_result) if agent_result is not None else None,
        fallback_used=fallback_used,
    )


st.set_page_config(page_title="RKLLM Chat UI (LangChain)", page_icon=":speech_balloon:", layout="wide")
init_state()

with st.sidebar:
    st.title("RKLLM Chat UI")
    st.session_state.api_base = st.text_input("API Base URL", value=st.session_state.api_base)
    st.session_state.model_name = st.text_input("Model", value=st.session_state.model_name)
    st.session_state.system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=120,
        help="Used as the LangChain agent system prompt.",
    )
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=float(st.session_state.temperature),
        step=0.1,
    )
    st.session_state.max_tokens = st.slider(
        "Max Tokens",
        min_value=32,
        max_value=2048,
        value=int(st.session_state.max_tokens),
        step=32,
    )
    st.session_state.timeout = st.slider(
        "Timeout (seconds)",
        min_value=10,
        max_value=600,
        value=int(st.session_state.timeout),
        step=10,
    )
    st.session_state.stream_responses = st.toggle("Stream Responses", value=st.session_state.stream_responses)
    st.session_state.show_trace_details = st.toggle("Show Trace Details", value=st.session_state.show_trace_details)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_messages = []
        st.session_state.last_trace = None
        st.rerun()

    with st.expander("Trace", expanded=False):
        trace_data = st.session_state.last_trace
        if trace_data:
            if st.session_state.show_trace_details:
                st.json(trace_data)
            else:
                st.json(
                    {
                        "request_id": trace_data.get("request_id"),
                        "api_base": trace_data.get("api_base"),
                        "model": trace_data.get("model"),
                        "mode": trace_data.get("mode"),
                        "status": trace_data.get("status"),
                        "error": trace_data.get("error"),
                        "message_count": len(trace_data.get("messages", [])),
                        "response_length": len(trace_data.get("response_text", "")),
                        "tool_runs": len(trace_data.get("tool_runs", [])),
                    }
                )
        else:
            st.caption("No trace captured yet.")

st.title("RKLLM OpenAI-Compatible Chat")
st.caption("Chat with the local RKLLM service through LangChain 1.x create_agent.")

render_history()

prompt = st.chat_input("Type a message and press Enter")
if prompt:
    request_id = f"ui-agent-{uuid.uuid4().hex}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.last_trace = {
        "request_id": request_id,
        "api_base": st.session_state.api_base,
        "model": st.session_state.model_name,
        "messages": list(st.session_state.agent_messages),
        "status": "pending",
        "error": None,
        "response_text": "",
        "tool_runs": [],
    }

    try:
        healthcheck(st.session_state.api_base, float(st.session_state.timeout))
        use_tooling = should_use_tooling(prompt)
        use_stream = bool(st.session_state.stream_responses and not use_tooling)
        st.session_state.last_trace["mode"] = "stream" if use_stream else "invoke"
        emit_ui_agent_request_trace(
            request_id,
            messages=list(st.session_state.agent_messages),
            mode=st.session_state.last_trace["mode"],
        )

        if use_stream:
            model = create_chat_model(request_id)
            with st.chat_message("assistant"):
                response_text = stream_plain_chat(model, list(st.session_state.agent_messages))
            result = {"streamed": True, "tools_enabled": False}
            tool_runs: list[dict[str, Any]] = []
        else:
            agent = create_langchain_agent(request_id)
            result = agent.invoke(
                {"messages": list(st.session_state.agent_messages)},
                config={"recursion_limit": 6},
            )
            response_text, tool_runs = normalize_agent_output(result)
            with st.chat_message("assistant"):
                think_text, answer_text = split_response_text(response_text)
                if think_text:
                    with st.expander("Thought Process", expanded=False):
                        st.markdown(think_text)
                st.markdown(answer_text or response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.agent_messages.append({"role": "assistant", "content": response_text})
        st.session_state.last_trace["status"] = "ok"
        st.session_state.last_trace["response_text"] = response_text
        st.session_state.last_trace["tool_runs"] = tool_runs
        st.session_state.last_trace["agent_result"] = str(result)
        emit_ui_agent_response_trace(
            request_id,
            mode=st.session_state.last_trace["mode"],
            status="ok",
            response_text=response_text,
            tool_runs=tool_runs,
            agent_result=result,
        )
    except httpx.HTTPError as exc:
        response_text = f"Request failed: {exc}"
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.last_trace["status"] = "http_error"
        st.session_state.last_trace["error"] = str(exc)
        st.session_state.last_trace["response_text"] = response_text
        emit_ui_agent_response_trace(
            request_id,
            mode=st.session_state.last_trace["mode"],
            status="http_error",
            response_text=response_text,
            error=str(exc),
        )
    except Exception as exc:
        error_text = str(exc)
        fallback_text = fallback_tool_answer(prompt) if "Recursion limit" in error_text else None
        response_text = fallback_text or f"Request failed: {exc}"
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.last_trace["status"] = "fallback" if fallback_text else "error"
        st.session_state.last_trace["error"] = error_text
        st.session_state.last_trace["response_text"] = response_text
        if fallback_text:
            st.session_state.last_trace["fallback_used"] = True
        emit_ui_agent_response_trace(
            request_id,
            mode=st.session_state.last_trace["mode"],
            status=st.session_state.last_trace["status"],
            response_text=response_text,
            error=error_text,
            fallback_used=bool(fallback_text),
        )

    st.rerun()
