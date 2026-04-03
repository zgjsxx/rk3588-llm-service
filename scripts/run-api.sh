#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${ROOT_DIR}/dist/native/linux_aarch64"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm}"
DEFAULT_MODEL_PATH="${ROOT_DIR}/models/${MODEL_NAME}"
MODEL_PATH="${RKLLM_MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
BRIDGE_LIB="${RKLLM_BRIDGE_LIB:-${INSTALL_DIR}/librkllm_openai_bridge.so}"
RUNTIME_LIB_DIR="${INSTALL_DIR}/lib"
TRACE_DIR="${RKLLM_TRACE_DIR:-${ROOT_DIR}/logs}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -f "${BRIDGE_LIB}" ]]; then
  echo "bridge library not found: ${BRIDGE_LIB}"
  echo "run ./build.sh first"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "model file not found: ${MODEL_PATH}"
  exit 1
fi

export RKLLM_MODEL_PATH="${MODEL_PATH}"
export RKLLM_BRIDGE_LIB="${BRIDGE_LIB}"
export LD_LIBRARY_PATH="${RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export RKLLM_MAX_NEW_TOKENS="${RKLLM_MAX_NEW_TOKENS:-2048}"
export RKLLM_MAX_CONTEXT_LEN="${RKLLM_MAX_CONTEXT_LEN:-2048}"
export RKLLM_TRACE="${RKLLM_TRACE:-1}"
export RKLLM_TRACE_DIR="${TRACE_DIR}"
export RKLLM_TRACE_FILE="${RKLLM_TRACE_FILE:-${TRACE_DIR}/native-trace.jsonl}"

cd "${ROOT_DIR}"
exec python -m uvicorn service.app:app --host "${HOST}" --port "${PORT}"
