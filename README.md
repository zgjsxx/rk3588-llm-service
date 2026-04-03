# DeepSeek-R1-Distill-Qwen-1.5B RKLLM Service

This repository packages the DeepSeek-R1-Distill-Qwen-1.5B RKLLM demo as:

- a native RKLLM bridge library in `native/`
- an OpenAI-compatible FastAPI service in `service/`
- a Streamlit chat UI in `streamlit_ui.py`
- one-command scripts in `scripts/`

## Repository Layout

```text
native/               C++ bridge and demo sources
service/              FastAPI service
scripts/              build and run entrypoints
models/               local .rkllm model files
third_party/          bundled RKLLM headers and runtime libraries
export/               model export helpers
tests/                Python tests
docs/                 additional documentation
```

## Requirements

```text
rkllm-runtime == 1.1.4
python == 3.8 or 3.10
```

## Python Virtual Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If `venv` is missing on Debian/Ubuntu:

```bash
sudo apt update
sudo apt install python3-venv
```

Install Python dependencies:

```bash
pip install -r requirements-fastapi.txt
```

Install with a domestic mirror if needed:

```bash
pip install -r requirements-fastapi.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

## Build

```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

Build output is written to:

```text
dist/native/linux_aarch64/
```

## Run API

```bash
chmod +x scripts/run-api.sh
./scripts/run-api.sh
```

Trace is enabled by default. Logs are written under:

```text
logs/trace-YYYYMMDD.jsonl
logs/native-trace.jsonl
```

You can disable trace if needed:

```bash
RKLLM_TRACE=0 ./scripts/run-api.sh
```

## Run UI

Start the API first, then:

```bash
chmod +x scripts/run-ui.sh
./scripts/run-ui.sh
```

Open `http://127.0.0.1:8501`.

The Streamlit sidebar includes a `Trace` panel showing:

- current `request_id`
- messages sent to FastAPI
- final rendered prompt
- request status and error summary

## Model Export

```bash
cd export
python generate_data_quant.py -m /path/to/DeepSeek-R1-Distill-Qwen-1.5B
python export_rkllm.py
```
