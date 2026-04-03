from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import streamlit as st
from service.prompting import build_prompt
from service.response_cleaning import split_response_text


DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_MODEL = "rkllm-local"
DEFAULT_SYSTEM_PROMPT = ""


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_base" not in st.session_state:
        st.session_state.api_base = DEFAULT_API_BASE
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "use_stream" not in st.session_state:
        st.session_state.use_stream = False
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.8
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.95
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 2048
    if "timeout" not in st.session_state:
        st.session_state.timeout = 120
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None
    if "show_trace_details" not in st.session_state:
        st.session_state.show_trace_details = False


def build_messages(system_prompt: str, current_prompt: str) -> list[dict[str, str]]:
    del system_prompt
    return [{"role": "user", "content": current_prompt}]


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            tool_calls = message.get("tool_calls")
            if tool_calls:
                st.markdown("Function call requested:")
                st.json(tool_calls)
                continue

            think_text, answer_text = split_response_text(message["content"])
            if think_text:
                with st.expander("Thought Process", expanded=False):
                    st.markdown(think_text)
            st.markdown(answer_text or message["content"])


def call_non_stream(payload: dict[str, Any], timeout: float) -> tuple[str, str, list[dict[str, Any]] | None]:
    response = httpx.post(
        f"{st.session_state.api_base}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls")
    return content, data.get("request_id", response.headers.get("X-Request-Id", "")), tool_calls


def stream_response(payload: dict[str, Any], timeout: float):
    with httpx.stream(
        "POST",
        f"{st.session_state.api_base}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        request_id = response.headers.get("X-Request-Id", "")
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue

            data = line[6:]
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content")
            if content:
                request_id = chunk.get("request_id", request_id)
                yield request_id, content


st.set_page_config(page_title="RKLLM Chat UI", page_icon=":speech_balloon:", layout="wide")
init_state()

with st.sidebar:
    st.title("RKLLM Chat UI")
    st.session_state.api_base = st.text_input("API Base URL", value=st.session_state.api_base)
    st.session_state.model_name = st.text_input("Model", value=st.session_state.model_name)
    st.session_state.system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=120,
        help="Ignored in llm_demo-compatible prompt mode. Only the latest user message is sent to the model.",
    )
    st.session_state.use_stream = st.toggle("Stream", value=st.session_state.use_stream)
    st.session_state.temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=float(st.session_state.temperature), step=0.1)
    st.session_state.top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=float(st.session_state.top_p), step=0.05)
    st.session_state.max_tokens = st.slider("Max Tokens", min_value=32, max_value=2048, value=int(st.session_state.max_tokens), step=32)
    st.session_state.timeout = st.slider("Timeout (seconds)", min_value=10, max_value=600, value=int(st.session_state.timeout), step=10)
    st.session_state.show_trace_details = st.toggle("Show Trace Details", value=st.session_state.show_trace_details)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.rerun()

    with st.expander("Trace", expanded=False):
        trace_data = st.session_state.last_trace
        if trace_data:
            if st.session_state.show_trace_details:
                st.json(trace_data)
            else:
                summary = {
                    "request_id": trace_data.get("request_id"),
                    "api_base": trace_data.get("api_base"),
                    "stream": trace_data.get("stream"),
                    "status": trace_data.get("status"),
                    "error": trace_data.get("error"),
                    "message_count": len(trace_data.get("messages", [])),
                    "prompt_length": len(trace_data.get("prompt", "")),
                    "response_length": len(trace_data.get("response_text", "")),
                }
                st.json(summary)
        else:
            st.caption("No trace captured yet.")

st.title("RKLLM OpenAI-Compatible Chat")
st.caption("Chat with the local RKLLM service through FastAPI `/v1/chat/completions`.")

render_history()

prompt = st.chat_input("Type a message and press Enter")
if prompt:
    request_id = f"ui-{uuid.uuid4().hex}"
    payload_messages = build_messages(st.session_state.system_prompt, prompt)
    rendered_prompt = build_prompt(
        payload_messages,
        prompt_prefix="<｜begin▁of▁sentence｜><｜User｜>",
        prompt_postfix="<｜Assistant｜>",
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "request_id": request_id,
        "model": st.session_state.model_name,
        "messages": payload_messages,
        "stream": st.session_state.use_stream,
        "max_tokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
    }
    st.session_state.last_trace = {
        "request_id": request_id,
        "api_base": st.session_state.api_base,
        "messages": payload_messages,
        "prompt": rendered_prompt,
        "stream": st.session_state.use_stream,
        "status": "pending",
        "error": None,
        "response_text": "",
    }

    with st.chat_message("assistant"):
        assistant_placeholder = st.empty()
        tool_calls: list[dict[str, Any]] | None = None
        try:
            if st.session_state.use_stream:
                full_text = ""
                backend_request_id = request_id
                for backend_request_id, token in stream_response(payload, st.session_state.timeout):
                    full_text += token
                    think_text, answer_text = split_response_text(full_text)
                    with assistant_placeholder.container():
                        if think_text:
                            with st.expander("Thought Process", expanded=False):
                                st.markdown(think_text)
                        st.markdown(answer_text or full_text)
                st.session_state.last_trace["request_id"] = backend_request_id
            else:
                full_text, backend_request_id, tool_calls = call_non_stream(payload, st.session_state.timeout)
                st.session_state.last_trace["request_id"] = backend_request_id or request_id
                with assistant_placeholder.container():
                    if tool_calls:
                        st.markdown("Function call requested:")
                        st.json(tool_calls)
                    else:
                        think_text, answer_text = split_response_text(full_text)
                        if think_text:
                            with st.expander("Thought Process", expanded=False):
                                st.markdown(think_text)
                        st.markdown(answer_text or full_text)
            st.session_state.last_trace["status"] = "ok"
            st.session_state.last_trace["response_text"] = full_text
            st.session_state.last_trace["tool_calls"] = tool_calls
        except httpx.HTTPStatusError as exc:
            try:
                error_payload = exc.response.json()
                message = error_payload.get("error", {}).get("message", exc.response.text)
            except Exception:
                message = exc.response.text
            full_text = f"Request failed: {message}"
            assistant_placeholder.error(full_text)
            st.session_state.last_trace["status"] = "http_error"
            st.session_state.last_trace["error"] = message
            st.session_state.last_trace["response_text"] = full_text
        except Exception as exc:
            full_text = f"Request failed: {exc}"
            assistant_placeholder.error(full_text)
            st.session_state.last_trace["status"] = "error"
            st.session_state.last_trace["error"] = str(exc)
            st.session_state.last_trace["response_text"] = full_text

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": full_text, "tool_calls": tool_calls})
