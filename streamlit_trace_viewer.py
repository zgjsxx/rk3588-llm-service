from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from service.trace import get_trace_dir


STEP_DEFINITIONS = [
    ("Agent 发起请求", "ui.agent.request"),
    ("Service 收到原始请求", "fastapi.request.payload"),
    ("Service 解析后的请求", "fastapi.request.received"),
    ("发送给模型的提示词", "fastapi.prompt.built"),
    ("模型原始输出", "fastapi.model.output"),
    ("Service 返回结果", "fastapi.response.payload"),
    ("Agent 最终展示结果", "ui.agent.response"),
]

SERVICE_MULTI_ROUND_STAGES = {
    "fastapi.request.payload",
    "fastapi.request.received",
    "fastapi.prompt.built",
    "fastapi.model.output",
    "fastapi.response.payload",
}


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _last_stage_event(events_by_stage: dict[str, list[dict[str, Any]]], stage: str) -> dict[str, Any] | None:
    values = events_by_stage.get(stage) or []
    return values[-1] if values else None


def list_trace_files() -> list[Path]:
    trace_dir = get_trace_dir()
    if not trace_dir.exists():
        return []
    return sorted(trace_dir.glob("trace-*.jsonl"), reverse=True)


def load_trace_events(trace_file: Path) -> list[dict[str, Any]]:
    if not trace_file.exists():
        return []
    events: list[dict[str, Any]] = []
    with trace_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def aggregate_requests(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    ungrouped_index = 0

    for event in events:
        request_id = event.get("request_id")
        if not request_id:
            request_id = f"ungrouped-{ungrouped_index}"
            ungrouped_index += 1

        bucket = grouped.setdefault(
            request_id,
            {
                "request_id": request_id,
                "events_by_stage": defaultdict(list),
                "all_events": [],
            },
        )
        bucket["all_events"].append(event)
        bucket["events_by_stage"][event.get("stage", "unknown")].append(event)

    requests: list[dict[str, Any]] = []
    for item in grouped.values():
        all_events = sorted(item["all_events"], key=lambda value: value.get("timestamp", ""))
        first_event = all_events[0] if all_events else {}
        last_event = all_events[-1] if all_events else {}
        started_at = _parse_timestamp(first_event.get("timestamp"))
        ended_at = _parse_timestamp(last_event.get("timestamp"))
        total_duration_ms = None
        if started_at and ended_at:
            total_duration_ms = round((ended_at - started_at).total_seconds() * 1000, 2)

        parsed_event = _last_stage_event(item["events_by_stage"], "fastapi.response.parsed")
        ui_response = _last_stage_event(item["events_by_stage"], "ui.agent.response")
        raw_request = _last_stage_event(item["events_by_stage"], "fastapi.request.payload")
        round_count = len(item["events_by_stage"].get("fastapi.prompt.built") or [])

        tool_used = bool(
            (parsed_event or {}).get("parsed_tool_calls")
            or (ui_response or {}).get("tool_runs")
            or ((raw_request or {}).get("raw_payload") or {}).get("tools")
        )
        status = (
            (ui_response or {}).get("status")
            or ("error" if any(event.get("error") for event in all_events) else "ok")
        )

        requests.append(
            {
                "request_id": item["request_id"],
                "events_by_stage": item["events_by_stage"],
                "all_events": all_events,
                "first_timestamp": first_event.get("timestamp"),
                "last_timestamp": last_event.get("timestamp"),
                "total_duration_ms": total_duration_ms,
                "tool_used": tool_used,
                "status": status,
                "round_count": round_count,
            }
        )

    return sorted(requests, key=lambda value: value.get("last_timestamp") or "", reverse=True)


def _step_payload(event: dict[str, Any], stage: str) -> Any:
    if stage == "ui.agent.request":
        return {
            "api_base": event.get("api_base"),
            "model": event.get("model"),
            "mode": event.get("mode"),
            "messages": event.get("messages"),
        }
    if stage == "fastapi.request.payload":
        return {
            "client": event.get("client"),
            "raw_payload": event.get("raw_payload"),
            "raw_payload_error": event.get("raw_payload_error"),
        }
    if stage == "fastapi.request.received":
        return {
            "model": event.get("model"),
            "stream": event.get("stream"),
            "messages": event.get("messages"),
            "tools": event.get("tools"),
            "tool_choice": event.get("tool_choice"),
            "client": event.get("client"),
        }
    if stage == "fastapi.prompt.built":
        return event.get("prompt")
    if stage == "fastapi.model.output":
        return event.get("output_text")
    if stage == "fastapi.response.payload":
        return event.get("response_payload")
    if stage == "ui.agent.response":
        return {
            "status": event.get("status"),
            "response_text": event.get("response_text"),
            "tool_runs": event.get("tool_runs"),
            "error": event.get("error"),
            "fallback_used": event.get("fallback_used"),
        }
    return event


def _render_payload(payload: Any) -> None:
    if isinstance(payload, (dict, list)):
        st.json(payload)
    else:
        st.code(str(payload or ""), language="text")


def _render_step(label: str, stage: str, events_by_stage: dict[str, list[dict[str, Any]]]) -> None:
    st.markdown(f"### {label}")
    st.caption(stage)
    events = events_by_stage.get(stage) or []
    if not events:
        st.info("该步骤无数据")
        return

    if stage not in SERVICE_MULTI_ROUND_STAGES:
        _render_payload(_step_payload(events[-1], stage))
        return

    for index, event in enumerate(events, start=1):
        timestamp = event.get("timestamp") or "-"
        with st.expander(f"第 {index} 轮 | {timestamp}", expanded=index == len(events)):
            _render_payload(_step_payload(event, stage))


def _render_summary(request_data: dict[str, Any]) -> None:
    st.subheader("本轮摘要")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Request ID", request_data["request_id"])
    col2.metric("状态", str(request_data.get("status") or "unknown"))
    col3.metric("调用工具", "是" if request_data.get("tool_used") else "否")
    duration = request_data.get("total_duration_ms")
    col4.metric("总耗时", f"{duration} ms" if duration is not None else "-")
    col5.metric("Service 轮次", str(request_data.get("round_count") or 0))


def _render_diagnostics(request_data: dict[str, Any]) -> None:
    st.subheader("异常与修复")
    parsed_event = _last_stage_event(request_data["events_by_stage"], "fastapi.response.parsed")
    repair_events = request_data["events_by_stage"].get("fastapi.response.repair") or []
    ui_response = _last_stage_event(request_data["events_by_stage"], "ui.agent.response")

    diagnostics = {
        "tool_call_parse_error": (parsed_event or {}).get("tool_call_parse_error"),
        "parsed_tool_calls": (parsed_event or {}).get("parsed_tool_calls"),
        "repair_events": repair_events,
        "tool_runs": (ui_response or {}).get("tool_runs"),
        "error": (ui_response or {}).get("error"),
        "fallback_used": (ui_response or {}).get("fallback_used"),
    }
    st.json(diagnostics)


st.set_page_config(page_title="RKLLM Trace Viewer", page_icon=":mag:", layout="wide")
st.title("RKLLM Trace Viewer")
st.caption("按 request_id 查看 agent -> service -> model -> service -> agent 的完整链路。")

trace_files = list_trace_files()
with st.sidebar:
    st.title("Trace 控制台")
    if not trace_files:
        st.warning("未找到 trace 文件。")
        selected_trace = None
    else:
        selected_trace = st.selectbox(
            "Trace 文件",
            options=trace_files,
            format_func=lambda path: path.name,
            index=0,
        )
    recent_limit = st.slider("最近请求数", min_value=10, max_value=500, value=100, step=10)
    status_filter = st.selectbox("状态筛选", ["全部", "ok", "error", "fallback", "http_error"])
    request_search = st.text_input("按 request_id 搜索", value="")
    if st.button("刷新", use_container_width=True):
        st.rerun()

if not trace_files:
    st.stop()

events = load_trace_events(selected_trace)
requests = aggregate_requests(events)

if request_search.strip():
    requests = [item for item in requests if request_search.strip() in item["request_id"]]
if status_filter != "全部":
    requests = [item for item in requests if item.get("status") == status_filter]
requests = requests[:recent_limit]

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("请求列表")
    if not requests:
        st.info("没有匹配的请求。")
    request_options = {
        f"{item['request_id']} | {item.get('status')} | {item.get('round_count', 0)} 轮 | {item.get('last_timestamp')}": item
        for item in requests
    }
    selected_label = (
        st.radio(
            "选择请求",
            options=list(request_options.keys()),
            label_visibility="collapsed",
        )
        if requests
        else None
    )

request_data = request_options[selected_label] if requests and selected_label else None

with right_col:
    if request_data is None:
        st.info("请先从左侧选择一条请求。")
    else:
        _render_summary(request_data)
        st.divider()
        for label, stage in STEP_DEFINITIONS:
            _render_step(label, stage, request_data["events_by_stage"])
            st.divider()
        _render_diagnostics(request_data)
