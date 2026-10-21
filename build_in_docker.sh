#!/usr/bin/env bash
set -euo pipefail

# Parse command line arguments
OS="${1:-ubuntu}"        # Default to Ubuntu
PRESET="${2:-linux-package-deb}"  # Default preset

mkdir -p oflm-build npu-cache

echo "Using OS: $OS, preset: $PRESET"

# Choose the correct Dockerfile based on OS
case "$OS" in
  "fedora")
    DOCKERFILE="Dockerfile.fedora"
    IMAGE="fedora"
    ;;
  "ubuntu")
    DOCKERFILE="Dockerfile"
    IMAGE="openflowlm-build:ubuntu26"
    ;;
  *)
    echo "Error: Invalid OS specified. Use 'ubuntu' or 'fedora'."
    exit 1
    ;;
esac

docker build \
  --target builder \
  -f "$DOCKERFILE" \
  -t "$IMAGE" \
  .

docker run --rm -it \
  --device=/dev/accel/accel0 \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  -e CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" \
  -e CTEST_PARALLEL_LEVEL="$(nproc)" \
  -e NPU_CACHE_HOME=/root/.npu/cache \
  -v "$PWD:/code:z" \
  -v "$PWD/npu-cache:/root/.npu/cache:z" \
  -v oflm-ccache:/var/cache/oflm-ccache \
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

