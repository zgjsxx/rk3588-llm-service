#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"

cd "${ROOT_DIR}"
exec streamlit run streamlit_ui.py --server.address "${HOST}" --server.port "${PORT}"
