#!/bin/bash
# build-all.sh — build flm, every open-kernel xclbin set, and install it all.
#
#   utilities/build-all.sh [--prefix DIR] [--specs a,b,...] [--skip-kernels]
#                          [--skip-app] [--harness] [--force] [--jobs N]
#
# What it does, in order:
#   1. Makes sure the kernel toolchain venv exists (ironvenv/, mlir-aie 1.4.2 +
#      Peano wheel from ironvenv-requirements.txt) and puts Peano + XRT tools
#      (xclbinutil, aiebu-asm) on PATH.
#   2. Runs open_kernels/export_qwen36_kernels.py for every recipe spec
#      (open_kernels/recipes/specs/*.json), building BOTH the q4nx kernel sets
#      and the GGUF-direct f32-scale twins into src/xclbins/<Model>/open_kernels.
#      A spec that fails does not stop the rest; a summary is printed at the end.
#   3. Configures + builds flm with the src/ linux-default preset (open engines
#      compiled in; the closed engine .so under src/lib/<runtime> install as-is).
#   4. cmake --install into --prefix (default: /opt/fastflowlm when writable,
#      otherwise ./install in the repo root — the same tree layout as the real
#      install: bin/flm, lib64/*.so, share/flm/{model_list.json,xclbins/...}).
#
# After it finishes, point flm at the install tree, e.g.:
#   PATH="$PWD/install/bin:$PATH" FLM_SHARE="$PWD/install/share/flm" flm run ...
# (or --prefix /opt/fastflowlm with sudo to update the system install).

set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
SPECS_DIR="$REPO/open_kernels/recipes/specs"
SRC="$REPO/src"
LOGDIR="$REPO/build-logs"
mkdir -p "$LOGDIR"

PREFIX=""
SPECS=""
SKIP_KERNELS=0
SKIP_APP=0
HARNESS=0
FORCE=""
JOBS="$(nproc 2>/dev/null || echo 4)"

usage() { grep '^#' "$0" | sed -n '2,12p'; exit 0; }
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --specs)  SPECS="$2"; shift 2 ;;
    --skip-kernels) SKIP_KERNELS=1; shift ;;
    --skip-app) SKIP_APP=1; shift ;;
    --harness) HARNESS=1; shift ;;
    --force) FORCE="--force"; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done

echo "== build-all: repo $REPO"

# ---- 1. kernel toolchain venv ------------------------------------------------
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

# ---- 2. the xclbin exports ---------------------------------------------------
failed_specs=()
ok_specs=()
if [ "$SKIP_KERNELS" -eq 0 ]; then
  if [ -n "$SPECS" ]; then
    spec_list=""
    IFS=',' read -ra _parts <<< "$SPECS"
    for s in "${_parts[@]}"; do
      f="$SPECS_DIR/$s"
      [ -f "$f" ] || f="$SPECS_DIR/$s.json"
      spec_list+=" $f"
    done
  else
    spec_list=$(ls "$SPECS_DIR"/*.json)
  fi
  for spec in $spec_list; do
    name="$(basename "$spec" .json)"
    echo "-- export $name"
    if "$VENV_PY" "$REPO/open_kernels/export_qwen36_kernels.py" --spec "$spec" $FORCE \
        >"$LOGDIR/export-$name.log" 2>&1; then
      ok_specs+=("$name")
      tail -1 "$LOGDIR/export-$name.log" | sed 's/^/   /'
    else
      failed_specs+=("$name")
      echo "   FAILED (see $LOGDIR/export-$name.log)"
    fi
  done
fi

# ---- 3. flm ------------------------------------------------------------------
if [ "$SKIP_APP" -eq 0 ]; then
  echo "-- cmake configure (src linux-default)"
  extra=()
  [ "$HARNESS" -eq 1 ] && extra+=(-DFLM_BUILD_OPEN_KERNELS_HARNESS=ON)
  cmake --preset linux-default "${extra[@]}" --log-level=WARNING -S "$SRC" || exit 1
  echo "-- cmake build (jobs=$JOBS)"
  cmake --build "$SRC/build" --parallel "$JOBS" || exit 1
fi

# ---- 4. install --------------------------------------------------------------
if [ -z "$PREFIX" ]; then
  if [ -w /opt ] || { [ -d /opt/fastflowlm ] && [ -w /opt/fastflowlm ]; }; then
    PREFIX=/opt/fastflowlm
  else
    PREFIX="$REPO/install"
  fi
fi
if [ "$SKIP_APP" -eq 0 ]; then
  echo "-- cmake --install -> $PREFIX"
  cmake --install "$SRC/build" --prefix "$PREFIX" >"$LOGDIR/install.log" 2>&1 || {
    tail -20 "$LOGDIR/install.log"; exit 1; }
fi

# ---- summary -----------------------------------------------------------------
echo
echo "== done"
[ "$SKIP_KERNELS" -eq 0 ] && echo "   kernels ok:   ${ok_specs[*]:-none}"
[ ${#failed_specs[@]} -gt 0 ] && echo "   kernels FAIL: ${failed_specs[*]} (logs in $LOGDIR/)"
if [ "$SKIP_APP" -eq 0 ]; then
  echo "   install tree:"
  du -sh "$PREFIX/bin/flm" "$PREFIX"/lib*/*.so 2>/dev/null | head -3 | sed 's/^/     /'
  ls "$PREFIX/share/flm/xclbins" 2>/dev/null | sed 's/^/     xclbins\//' | head -12
  for d in "$PREFIX/share/flm/xclbins"/*/open_kernels; do
    [ -d "$d" ] && echo "     xclbins/$(basename "$(dirname "$d")")/open_kernels: $(ls "$d" | head -1 | xargs -I{} echo {}) ... (manifest: $(test -f "$d/manifest.json" && echo yes || echo MISSING))"
  done
fi
[ ${#failed_specs[@]} -gt 0 ] && exit 1
exit 0
