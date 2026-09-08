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
#   4. cmake --install into --prefix (default /opt/fastflowlm; sudo is used for
#      the install step itself when the prefix needs root — the toolchain steps
#      stay unprivileged). The install tree matches the real one: bin/flm,
#      lib64/*.so, share/flm/{model_list.json,xclbins/...}.
#
# Root is requested at most twice, and only interactively: once if a past
# sudo'd build left root-owned artifacts under the repo (step 0 heals them),
# and once for the install. Never run this whole script under sudo — the venv
# toolchain ends up root-owned and breaks.
#
# After it finishes, point flm at the install tree, e.g.:
#   PATH="$PWD/install/bin:$PATH" FLM_SHARE="$PWD/install/share/flm" flm run ...

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

# ---- 0. heal root-owned build artifacts --------------------------------------
# A past sudo'd build can leave root-owned files (e.g. fetched third_party
# deps); plain user cmake then fails with "Error removing directory". Ask for
# root ONCE here (this is the only other privileged moment besides install).
_roots=$(find "$REPO/third_party" "$REPO/src/build" "$REPO/install" -user root 2>/dev/null | head -1)
if [ -n "$_roots" ]; then
  echo "-- found root-owned build artifacts (from a past sudo'd build); fixing"
  if command -v sudo >/dev/null 2>&1; then
    sudo -n chown -R "$(id -un):$(id -gn)" "$REPO/third_party" "$REPO/src/build" "$REPO/install" 2>/dev/null \
      || sudo chown -R "$(id -un):$(id -gn)" "$REPO/third_party" "$REPO/src/build" "$REPO/install" || {
        echo "   FATAL: could not chown root-owned artifacts; run manually:" >&2
        echo "     sudo chown -R $(id -un):$(id -gn) $REPO/third_party $REPO/src/build $REPO/install" >&2
        exit 1
      }
  else
    echo "   FATAL: no sudo; run manually:" >&2
    echo "     sudo chown -R $(id -un):$(id -gn) $REPO/third_party $REPO/src/build $REPO/install" >&2
    exit 1
  fi
fi

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
# Default to the real system install (/opt/fastflowlm); only fall back to a
# local staging tree if sudo is refused (running the WHOLE script under sudo
# breaks the venv toolchain, so root is requested for this step only).
if [ -z "$PREFIX" ]; then
  PREFIX=/opt/fastflowlm
fi
if [ "$SKIP_APP" -eq 0 ]; then
  echo "-- cmake --install -> $PREFIX"
  if [ -w "$(dirname "$PREFIX")" ] || { [ -d "$PREFIX" ] && [ -w "$PREFIX" ]; }; then
    cmake --install "$SRC/build" --prefix "$PREFIX" >"$LOGDIR/install.log" 2>&1 || {
      tail -20 "$LOGDIR/install.log"; exit 1; }
  elif command -v sudo >/dev/null 2>&1; then
    echo "   need root for $PREFIX (sudo; enter password if prompted)"
    sudo cmake --install "$SRC/build" --prefix "$PREFIX" >"$LOGDIR/install.log" 2>&1 || {
      tail -20 "$LOGDIR/install.log"
      echo "   sudo install failed; retrying into a local tree"
      PREFIX="$REPO/install"
      cmake --install "$SRC/build" --prefix "$PREFIX" >>"$LOGDIR/install.log" 2>&1 || {
        tail -20 "$LOGDIR/install.log"; exit 1; }
    }
  else
    echo "   no sudo available; installing into a local tree"
    PREFIX="$REPO/install"
    cmake --install "$SRC/build" --prefix "$PREFIX" >"$LOGDIR/install.log" 2>&1 || {
      tail -20 "$LOGDIR/install.log"; exit 1; }
  fi
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
