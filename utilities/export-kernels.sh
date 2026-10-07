#!/usr/bin/env bash
# export-kernels.sh — build every open NPU kernel xclbin set so `cmake --install`
# ships a full distribution. Driven by the `export_kernels` CMake target
# (FLM_BUILD_KERNELS=ON). Safe to run repeatedly: each spec's build cache
# (build_key) skips specs already exported on the same toolchain.
#
# Usage: utilities/export-kernels.sh [--specs a,b,c] [--jobs N]
#
# Requirements (assumed present, or set up here):
#   * XRT installed at /opt/xilinx/xrt (xclbinutil on PATH).
#   * ironvenv/ created here from ironvenv-requirements.txt (mlir-aie + Peano).
#   * third_party/mlir-aie and third_party/Peano cloned here (best-effort; only
#     used for toolchain.json version metadata).

set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
SPECS_DIR="$REPO/open_kernels/recipes/specs"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
SPECS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --specs=*) SPECS="${1#*=}"; shift ;;
    --specs)   SPECS="$2"; shift 2 ;;
    --jobs=*)  JOBS="${1#*=}"; shift ;;
    --jobs)    JOBS="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# ---- 1. kernel toolchain venv ----------------------------------------------
if [ ! -x "$REPO/ironvenv/bin/python" ]; then
  echo "-- creating ironvenv (mlir-aie + Peano toolchain)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.13 "$REPO/ironvenv" || exit 1
    uv pip install --python "$REPO/ironvenv/bin/python" -r "$REPO/ironvenv-requirements.txt" || exit 1
  else
    python3 -m venv "$REPO/ironvenv" || exit 1
    "$REPO/ironvenv/bin/pip" install -r "$REPO/ironvenv-requirements.txt" || exit 1
  fi
fi
VENV_PY="$REPO/ironvenv/bin/python"
PEANO_DIR="$REPO/ironvenv/lib/python3.13/site-packages/llvm-aie/bin"
export PATH="$PEANO_DIR:/opt/xilinx/xrt/bin:$PATH"
for tool in clang xclbinutil aiebu-asm; do
  command -v "$tool" >/dev/null 2>&1 || { echo "FATAL: $tool not on PATH (Peano/XRT)" >&2; exit 1; }
done

# ---- 2. best-effort clone of third_party toolchain sources ------------------
if [ ! -d "$REPO/third_party/mlir-aie" ]; then
  echo "-- cloning third_party/mlir-aie (best-effort)"
  git clone --depth 1 https://github.com/Xilinx/mlir-aie "$REPO/third_party/mlir-aie" 2>/dev/null \
    || echo "   warn: mlir-aie clone failed (non-fatal)"
fi
if [ ! -d "$REPO/third_party/Peano" ]; then
  echo "-- cloning third_party/Peano (best-effort)"
  git clone --depth 1 https://gitlab.lrz.de/hpcsoftware/Peano.git "$REPO/third_party/Peano" 2>/dev/null \
    || echo "   warn: Peano clone failed (non-fatal)"
fi
[ -d "$REPO/third_party/mlir-aie" ] && export MLIR_AIE_ROOT="$REPO/third_party/mlir-aie"

# ---- 3. the xclbin exports --------------------------------------------------
if [ -n "$SPECS" ]; then
  spec_list=""
  IFS=',' read -ra _parts <<< "$SPECS"
  for s in "${_parts[@]}"; do
    f="$SPECS_DIR/$s"; [ -f "$f" ] || f="$SPECS_DIR/$s.json"; spec_list+=" $f"
  done
else
  spec_list=$(ls "$SPECS_DIR"/*.json)
fi
rc=0
for spec in $spec_list; do
  name="$(basename "$spec" .json)"
  echo "-- export $name"
  if "$VENV_PY" "$REPO/open_kernels/export_qwen36_kernels.py" --spec "$spec"; then
    echo "   ok: $name"
  else
    echo "   FAILED: $name"
    rc=1
  fi
done
exit $rc
