# RKLLM OpenAI-Compatible FastAPI Service

## Overview

This repository now includes:

- `native/src/rkllm_bridge.cpp`: a C ABI bridge over RKLLM for Python `ctypes`
- `service/app.py`: a FastAPI app exposing `/v1/chat/completions`
- `tests/`: protocol and prompt assembly tests

## Build

The project now assumes `third_party/rkllm_api/` lives in the repository root.

```bash
chmod +x scripts/build.sh scripts/run-api.sh scripts/run-ui.sh
./scripts/build.sh
```

If you want to override the local RKLLM SDK path or model path:

```bash
RKLLM_API_PATH=/path/to/rkllm_api MODEL_SRC=/path/to/model.rkllm ./scripts/build.sh
```

The resulting shared library is installed beside the demo binary:

```bash
dist/native/linux_aarch64/librkllm_openai_bridge.so
```

## Run the FastAPI service

Install dependencies:

```bash
pip install -r requirements-fastapi.txt
```

Start the API:

```bash
./scripts/run-api.sh
```

Trace is enabled by default. Python and native trace files are written to:

```text
logs/trace-YYYYMMDD.jsonl
logs/native-trace.jsonl
```

Useful debug endpoint:

```bash
curl http://127.0.0.1:8000/debug/trace
```

Optional overrides:

```bash
HOST=0.0.0.0 PORT=8000 ./scripts/run-api.sh
```

## Run the Streamlit UI

Start the backend first, then launch the web UI:

```bash
./scripts/run-ui.sh
```

Open:

```bash
http://127.0.0.1:8501
```

The UI entry file is `streamlit_ui.py`.

The Streamlit sidebar also shows the current request trace summary.

Optional override:

```bash
HOST=0.0.0.0 PORT=8501 ./scripts/run-ui.sh
```

## Example request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-local",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```
