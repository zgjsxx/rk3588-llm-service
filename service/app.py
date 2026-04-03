from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .prompting import build_prompt
from .rkllm_bridge import BridgeError, RkllmEngine, load_engine_config_from_env
from .schemas import ChatCompletionError, ChatCompletionErrorResponse, ChatCompletionRequest
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
                "fastapi.response.ready",
                request_id=request_id,
                duration_ms=round((time.time() - started_at) * 1000, 2),
                output_length=len(content),
                response_text=content,
            )

            response_payload = {
                "request_id": request_id,
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
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
