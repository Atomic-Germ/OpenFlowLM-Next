#!/usr/bin/env bash
set -euo pipefail

IMAGE="openflowlm-build:ubuntu26"
PRESET="${1:-linux-package}"

mkdir -p oflm-build npu-cache

echo "Using preset: $PRESET"

docker build \
  -f Dockerfile \
  -t "$IMAGE" \
  .

docker run --rm -it \
  --device=/dev/accel/accel0 \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  -e CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
  -e CTEST_PARALLEL_LEVEL="$(nproc)" \
  -e NPU_CACHE_HOME=/root/.npu/cache \
  -v "$PWD/build:/code/build" \
  -v "$PWD/npu-cache:/root/.npu/cache" \
  "$IMAGE" \
  cmake --workflow --preset "$PRESET"