#!/bin/bash
set -euo pipefail

BUILD_TYPE="${BUILD_TYPE:-Release}"
TARGET_ARCH="${TARGET_ARCH:-aarch64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NATIVE_DIR="${ROOT_DIR}/native"
BUILD_DIR="${ROOT_DIR}/build/native/linux_${TARGET_ARCH}_${BUILD_TYPE}"
INSTALL_DIR="${ROOT_DIR}/dist/native/linux_${TARGET_ARCH}"
RKLLM_API_PATH="${RKLLM_API_PATH:-${ROOT_DIR}/third_party/rkllm_api}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm}"
MODEL_SRC="${MODEL_SRC:-${ROOT_DIR}/models/${MODEL_NAME}}"

if [[ ! -f "${RKLLM_API_PATH}/include/rkllm.h" ]]; then
  echo "rkllm.h not found: ${RKLLM_API_PATH}/include/rkllm.h"
  exit 1
fi

if [[ ! -f "${RKLLM_API_PATH}/${TARGET_ARCH}/librkllmrt.so" ]]; then
  echo "librkllmrt.so not found: ${RKLLM_API_PATH}/${TARGET_ARCH}/librkllmrt.so"
  exit 1
fi

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"
cmake "${NATIVE_DIR}" \
  -DCMAKE_SYSTEM_PROCESSOR="${TARGET_ARCH}" \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DRKLLM_API_PATH="${RKLLM_API_PATH}"

make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
make install

if [[ -f "${MODEL_SRC}" ]]; then
  cp -f "${MODEL_SRC}" "${INSTALL_DIR}/${MODEL_NAME}"
  sync
else
  echo "warning: model file not found, skipped copy: ${MODEL_SRC}"
fi

echo "build complete"
echo "install dir: ${INSTALL_DIR}"
