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
#   3. Configures + builds oflm with the src/ linux-default preset (open engines
#      compiled in; the closed engine .so under src/lib/<runtime> install as-is).
#   4. cmake --install into --prefix (default /opt/openflowlm; sudo is used for
#      the install step itself when the prefix needs root — the toolchain steps
#      stay unprivileged). The install tree matches the real one: bin/oflm,
#      lib64/*.so, share/oflm/{model_list.json,xclbins/...}.
#
# Root is requested at most twice, and only interactively: once if a past
# sudo'd build left root-owned artifacts under the repo (step 0 heals them),
# and once for the install. Never run this whole script under sudo — the venv
# toolchain ends up root-owned and breaks.
#
# After it finishes, point oflm at the install tree, e.g.:
#   PATH="$PWD/install/bin:$PATH" OFLM_XCLBIN_PATH="$PWD/install/share/oflm/xclbins" oflm run ...

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
# deps, or the kernel-export step's final.prj trees under open_kernels/designs),
# and then plain-user cmake / build_design.py fails with "Permission denied" on
# the .lock file. Ask for root ONCE here (this is the only other privileged
# moment besides install).
_roots=$(find "$REPO/third_party" "$REPO/src/build" "$REPO/install" "$REPO/open_kernels" -user root 2>/dev/null | head -1)
if [ -n "$_roots" ]; then
  echo "-- found root-owned build artifacts (from a past sudo'd build); fixing"
  if command -v sudo >/dev/null 2>&1; then
    sudo -n chown -R "$(id -un):$(id -gn)" "$REPO/third_party" "$REPO/src/build" "$REPO/install" "$REPO/open_kernels" 2>/dev/null \
      || sudo chown -R "$(id -un):$(id -gn)" "$REPO/third_party" "$REPO/src/build" "$REPO/install" "$REPO/open_kernels" || {
        echo "   FATAL: could not chown root-owned artifacts; run manually:" >&2
        echo "     sudo chown -R $(id -un):$(id -gn) $REPO/third_party $REPO/src/build $REPO/install $REPO/open_kernels" >&2
        exit 1
      }
  else
    echo "   FATAL: no sudo; run manually:" >&2
    echo "     sudo chown -R $(id -un):$(id -gn) $REPO/third_party $REPO/src/build $REPO/install $REPO/open_kernels" >&2
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
# Derive site-packages from the interpreter that is actually there rather than
# assuming 3.13: an existing 3.12 ironvenv is a supported layout (and the one
# this repo ships), and a hard-coded 3.13 path left it valid here yet sent the
# tool checks at the end of this block hunting a Peano that was never on PATH.
PEANO_DIR="$("$VENV_PY" -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "llvm-aie", "bin"))')"
if [ ! -d "$PEANO_DIR" ]; then
  echo "FATAL: Peano (llvm-aie) not found at $PEANO_DIR" >&2
  echo "  Recreate the venv, or install it with:" >&2
  echo "    uv pip install --python $VENV_PY -r $REPO/ironvenv-requirements.txt" >&2
  exit 1
fi
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
  # src/CMakeLists.txt declares OFLM_BUILD_OPEN_KERNELS_HARNESS; the old FLM_ name
  # was silently ignored, so --harness did nothing.
  [ "$HARNESS" -eq 1 ] && extra+=(-DOFLM_BUILD_OPEN_KERNELS_HARNESS=ON)
  cmake --preset linux-default "${extra[@]}" --log-level=WARNING -S "$SRC" || exit 1
  echo "-- cmake build (jobs=$JOBS)"
  cmake --build "$SRC/build" --parallel "$JOBS" || exit 1
fi

# ---- 4. install --------------------------------------------------------------
# Default to the real system install. src/CMakePresets.json puts CMAKE_INSTALL_PREFIX
# at /opt/openflowlm and CMAKE_XCLBIN_PREFIX at /opt/openflowlm/share/oflm -- matching
# what the engine looks for ($OFLM_XCLBIN_PATH, utils.cpp:218). This script used to
# default to the pre-rename /opt/fastflowlm, which put the xclbins where the engine
# did not look. Only fall back to a local staging tree if sudo is refused (running
# the WHOLE script under sudo breaks the venv toolchain, so root is requested for
# this step only).
if [ -z "$PREFIX" ]; then
  PREFIX=/opt/openflowlm
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
  du -sh "$PREFIX/bin/oflm" "$PREFIX"/lib*/*.so 2>/dev/null | head -3 | sed 's/^/     /'
  ls "$PREFIX/share/oflm/xclbins" 2>/dev/null | sed 's/^/     xclbins\//' | head -12
  for d in "$PREFIX/share/oflm/xclbins"/*/open_kernels; do
    [ -d "$d" ] && echo "     xclbins/$(basename "$(dirname "$d")")/open_kernels: $(ls "$d" | head -1 | xargs -I{} echo {}) ... (manifest: $(test -f "$d/manifest.json" && echo yes || echo MISSING))"
  done
fi
[ ${#failed_specs[@]} -gt 0 ] && exit 1
exit 0
