#!/usr/bin/env bash
set -euo pipefail

IMAGE="openflowlm-build:ubuntu26"
PRESET="${1:-linux-package-deb}"

mkdir -p oflm-build npu-cache

echo "Using preset: $PRESET"

docker build \
  --target builder \
  -f Dockerfile \
  -t "$IMAGE" \
  .

docker run --rm -it \
  --device=/dev/accel/accel0 \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  -e CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
  -e CTEST_PARALLEL_LEVEL="$(nproc)" \
  -e LDFLAGS="-fuse-ld=lld" \
  -e NPU_CACHE_HOME=/root/.npu/cache \
  -v "$PWD:/code" \
  -v "$PWD/npu-cache:/root/.npu/cache" \
  "$IMAGE" \
  bash -eu -c '
    case "$1" in
      linux-package-deb|linux-package-rpm|linux-package-tgz)
        cmake --workflow --preset linux-default
        cpack --preset "$1"
        ;;
      *) cmake --workflow --preset "$1" ;;
    esac
  ' -- "$PRESET"
