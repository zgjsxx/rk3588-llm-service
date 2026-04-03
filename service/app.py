from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .prompting import build_prompt
from .rkllm_bridge import BridgeError, RkllmEngine, load_engine_config_from_env
from .schemas import ChatCompletionError, ChatCompletionErrorResponse, ChatCompletionRequest, ToolCall
from .trace import get_trace_file, trace_event


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _usage(prompt: str, completion: str) -> dict[str, int]:
    prompt_tokens = len(prompt.split())
    completion_tokens = len(completion.split())
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _extract_json_text(content: str) -> str:
    stripped = _THINK_TAG_PATTERN.sub("", content).strip()
    stripped = _CODE_FENCE_PATTERN.sub("", stripped).strip()
    return stripped


def _normalize_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


def _has_tool(request: ChatCompletionRequest, name: str) -> bool:
    return bool(request.tools) and any(tool.function.name == name for tool in request.tools)


def _extract_two_numbers(text: str) -> tuple[int | float, int | float] | None:
    matches = _NUMBER_PATTERN.findall(text)
    if len(matches) != 2:
        return None

    values: list[int | float] = []
    for match in matches:
        if "." in match:
            values.append(float(match))
        else:
            values.append(int(match))
    return values[0], values[1]


def _repair_add_arguments(arguments: Any) -> dict[str, Any] | None:
    if isinstance(arguments, dict):
        if "a" in arguments and "b" in arguments:
            return {"a": arguments["a"], "b": arguments["b"]}
        return None
    if isinstance(arguments, list):
        if len(arguments) == 2:
            return {"a": arguments[0], "b": arguments[1]}
        return None
    if isinstance(arguments, str):
        pair = _extract_two_numbers(arguments)
        if pair is None:
            return None
        return {"a": pair[0], "b": pair[1]}
    return None


def _repair_add_tool_payload(content: str) -> tuple[list[dict[str, Any]] | None, str | None, str | None]:
    cleaned = _extract_json_text(content)
    repaired_variants = [cleaned]
    if cleaned.count("{") > cleaned.count("}"):
        repaired_variants.append(cleaned + ("}" * (cleaned.count("{") - cleaned.count("}"))))
    if cleaned.count("[") > cleaned.count("]"):
        repaired_variants.append(cleaned + ("]" * (cleaned.count("[") - cleaned.count("]"))))

    for variant in repaired_variants:
        try:
            payload = json.loads(variant)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        raw_tool_calls = payload.get("tool_calls")
        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            continue
        first = raw_tool_calls[0]
        if not isinstance(first, dict):
            continue
        if first.get("type") == "function" and isinstance(first.get("function"), dict):
            function_payload = first["function"]
            name = function_payload.get("name")
            arguments = function_payload.get("arguments")
        else:
            name = first.get("name")
            arguments = first.get("arguments")
        if name != "add":
            continue
        repaired_arguments = _repair_add_arguments(arguments)
        if repaired_arguments is None:
            continue
        return (
            [
                ToolCall(
                    id=_tool_call_id(),
                    type="function",
                    function={
                        "name": "add",
                        "arguments": json.dumps(repaired_arguments, ensure_ascii=False, sort_keys=True),
                    },
                ).model_dump()
            ],
            "repaired_add_payload",
            None,
        )

    pair = _extract_two_numbers(cleaned)
    if pair is not None and "add" in cleaned:
        repaired_arguments = {"a": pair[0], "b": pair[1]}
        return (
            [
                ToolCall(
                    id=_tool_call_id(),
                    type="function",
                    function={
                        "name": "add",
                        "arguments": json.dumps(repaired_arguments, ensure_ascii=False, sort_keys=True),
                    },
                ).model_dump()
            ],
            "repaired_add_numbers",
            None,
        )
    return None, None, "repair_failed"


def _parse_single_tool_call(
    item: dict[str, Any],
    allowed_tool_names: set[str],
    forced_tool_name: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if item.get("type") == "function" and isinstance(item.get("function"), dict):
        function_payload = item["function"]
        name = function_payload.get("name")
        arguments = function_payload.get("arguments")
    else:
        name = item.get("name")
        arguments = item.get("arguments")

    if not isinstance(name, str) or not name:
        return None, "missing_function_name"
    if name not in allowed_tool_names:
        return None, f"unknown_function:{name}"
    if forced_tool_name is not None and name != forced_tool_name:
        return None, f"unexpected_function:{name}"
    if arguments is None:
        return None, f"missing_arguments:{name}"
    if not isinstance(arguments, (dict, str)):
        return None, f"invalid_arguments_type:{name}"

    tool_call = ToolCall(
        id=_tool_call_id(),
        type="function",
        function={
            "name": name,
            "arguments": _normalize_tool_arguments(arguments),
        },
    )
    return tool_call.model_dump(), None


def _parse_tool_calls(
    content: str,
    request: ChatCompletionRequest,
    request_id: str | None = None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    repair_strategy = None
    if not request.tools:
        return None, None
    if request.tool_choice == "none":
        return None, "tool_choice_none"

    try:
        payload = json.loads(_extract_json_text(content))
    except json.JSONDecodeError as exc:
        if _has_tool(request, "add"):
            repaired, repair_strategy, repair_error = _repair_add_tool_payload(content)
            if repaired is not None:
                trace_event(
                    "fastapi.response.repair",
                    request_id=request_id,
                    repair_applied=True,
                    repair_strategy=repair_strategy,
                    repair_error=None,
                    repaired_tool_calls=repaired,
                )
                return repaired, None
            trace_event(
                "fastapi.response.repair",
                request_id=request_id,
                repair_applied=False,
                repair_strategy=repair_strategy,
                repair_error=repair_error or f"invalid_json:{exc.msg}",
                repaired_tool_calls=None,
            )
        return None, f"invalid_json:{exc.msg}"

    if not isinstance(payload, dict):
        return None, "payload_not_object"

    raw_tool_calls = payload.get("tool_calls")
    if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
        return None, "missing_tool_calls"

    allowed_tool_names = {tool.function.name for tool in request.tools}
    forced_tool_name = None
    if request.tool_choice is not None and not isinstance(request.tool_choice, str):
        forced_tool_name = request.tool_choice.function.name

    parsed_tool_calls: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            return None, "tool_call_not_object"
        parsed_tool_call, parse_error = _parse_single_tool_call(item, allowed_tool_names, forced_tool_name)
        if parse_error is not None:
            if _has_tool(request, "add"):
                repaired, repair_strategy, repair_error = _repair_add_tool_payload(content)
                if repaired is not None:
                    trace_event(
                        "fastapi.response.repair",
                        request_id=request_id,
                        repair_applied=True,
                        repair_strategy=repair_strategy,
                        repair_error=None,
                        repaired_tool_calls=repaired,
                    )
                    return repaired, None
                trace_event(
                    "fastapi.response.repair",
                    request_id=request_id,
                    repair_applied=False,
                    repair_strategy=repair_strategy,
                    repair_error=repair_error or parse_error,
                    repaired_tool_calls=None,
                )
            return None, parse_error
        parsed_tool_calls.append(parsed_tool_call)

    return parsed_tool_calls, None


def create_app(engine_factory: Optional[Callable[[], RkllmEngine]] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.generation_lock = asyncio.Lock()
        factory = engine_factory or (lambda: RkllmEngine(load_engine_config_from_env()))
        app.state.engine = factory()
        try:
            yield
        finally:
            app.state.engine.close()

    app = FastAPI(title="RKLLM OpenAI Compatible API", lifespan=lifespan)

    @app.exception_handler(BridgeError)
    async def bridge_error_handler(_: Request, exc: BridgeError) -> JSONResponse:
        payload = ChatCompletionErrorResponse(
            error=ChatCompletionError(message=str(exc), type="server_error", code="rkllm_error")
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        payload = ChatCompletionErrorResponse(
            error=ChatCompletionError(
                message=str(exc.detail),
                type="invalid_request_error" if exc.status_code < 500 else "server_error",
                code="http_error",
            )
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        payload = ChatCompletionErrorResponse(
            error=ChatCompletionError(
                message=str(exc),
                type="invalid_request_error",
                code="validation_error",
            )
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/debug/trace")
    async def debug_trace(limit: int = 200) -> dict[str, object]:
        trace_file = get_trace_file()
        if not trace_file.exists():
            return {"events": [], "trace_file": str(trace_file)}

        with trace_file.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()[-limit:]

        return {
            "trace_file": str(trace_file),
            "events": [json.loads(line) for line in lines],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        engine = app.state.engine
        request_id = request.request_id or _completion_id()
        trace_event(
            "fastapi.request.received",
            request_id=request_id,
            model=request.model,
            stream=request.stream,
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            client=str(raw_request.client),
        )
        prompt = build_prompt(
            request.messages,
            prompt_prefix=getattr(engine.config, "prompt_prefix", "<|begin_of_sentence|><|User|>"),
            prompt_postfix=getattr(engine.config, "prompt_postfix", "<|Assistant|>"),
            tools=request.tools,
            tool_choice=request.tool_choice,
        )
        trace_event(
            "fastapi.prompt.built",
            request_id=request_id,
            prompt=prompt,
            prompt_length=len(prompt),
            prompt_lines=len(prompt.splitlines()),
        )

        if request.model != getattr(engine.config, "model_name", request.model):
            raise HTTPException(status_code=400, detail=f"Unsupported model '{request.model}'")

        if request.stream and request.tools:
            raise HTTPException(status_code=400, detail="Streaming is not supported for tool calls yet")

        created = int(time.time())
        completion_id = request_id

        if not request.stream:
            started_at = time.time()
            async with app.state.generation_lock:
                content = await asyncio.to_thread(
                    engine.generate,
                    prompt,
                    lambda _: None,
                    request_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                )
            trace_event(
                "fastapi.model.output",
                request_id=request_id,
                output_length=len(content),
                output_text=content,
            )
            trace_event(
                "fastapi.response.ready",
                request_id=request_id,
                duration_ms=round((time.time() - started_at) * 1000, 2),
                output_length=len(content),
                response_text=content,
            )

            parsed_tool_calls, parse_error = _parse_tool_calls(content, request, request_id=request_id)
            trace_event(
                "fastapi.response.parsed",
                request_id=request_id,
                parsed_tool_calls=parsed_tool_calls,
                tool_call_parse_error=parse_error,
            )

            message: dict[str, Any] = {"role": "assistant", "content": content}
            finish_reason = "stop"
            if parsed_tool_calls is not None:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": parsed_tool_calls,
                }
                finish_reason = "tool_calls"

            response_payload = {
                "request_id": request_id,
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": _usage(prompt, content),
            }
            response = JSONResponse(content=response_payload)
            response.headers["X-Request-Id"] = request_id
            return response

        async def event_stream() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
            error_holder: dict[str, str] = {}
            saw_first_token = False
            full_text = ""

            def on_token(token: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, token)

            async with app.state.generation_lock:
                trace_event("fastapi.stream.start", request_id=request_id)
                worker = asyncio.create_task(
                    asyncio.to_thread(
                        engine.generate,
                        prompt,
                        on_token,
                        request_id,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        presence_penalty=request.presence_penalty,
                        frequency_penalty=request.frequency_penalty,
                    )
                )

                try:
                    while True:
                        if await raw_request.is_disconnected():
                            trace_event("fastapi.stream.client_disconnected", request_id=request_id)
                            await asyncio.to_thread(engine.cancel)
                            worker.cancel()
                            break

                        if worker.done() and queue.empty():
                            try:
                                await worker
                            except BridgeError as exc:
                                error_holder["message"] = str(exc)
                            break

                        try:
                            token = await asyncio.wait_for(queue.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            continue
                        if not saw_first_token:
                            saw_first_token = True
                            trace_event("fastapi.stream.first_token", request_id=request_id)
                        full_text += token

                        chunk = {
                            "request_id": request_id,
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    if error_holder:
                        trace_event("fastapi.stream.error", request_id=request_id, error=error_holder["message"])
                        raise BridgeError(error_holder["message"])

                    trace_event(
                        "fastapi.model.output",
                        request_id=request_id,
                        stream=True,
                        output_length=len(full_text),
                        output_text=full_text,
                    )
                    trace_event(
                        "fastapi.stream.finish",
                        request_id=request_id,
                        output_length=len(full_text),
                        response_text=full_text,
                    )
                    final_chunk = {
                        "request_id": request_id,
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    if not worker.done():
                        worker.cancel()

        response = StreamingResponse(event_stream(), media_type="text/event-stream")
        response.headers["X-Request-Id"] = request_id
        return response

    return app


app = create_app()
