from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


_TRACE_LOCK = Lock()


def trace_enabled() -> bool:
    return os.getenv("RKLLM_TRACE", "1") != "0"


def get_trace_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    return Path(os.getenv("RKLLM_TRACE_DIR", str(root / "logs")))


def get_trace_file() -> Path:
    trace_dir = get_trace_dir()
    filename = f"trace-{datetime.now().strftime('%Y%m%d')}.jsonl"
    return trace_dir / filename


def serialize_trace_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Exception):
        return {"type": value.__class__.__name__, "message": str(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): serialize_trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_trace_value(item) for item in value]
    return str(value)


def trace_event(stage: str, **fields: Any) -> None:
    if not trace_enabled():
        return

    trace_file = get_trace_file()
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "stage": stage,
    }
    event.update({key: serialize_trace_value(value) for key, value in fields.items()})

    with _TRACE_LOCK:
        with trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
