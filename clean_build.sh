#!/usr/bin/env bash
#
# clean_build.sh — Build flm from a clean CMake tree on Linux.
#
# Linux counterpart of clean_build.bat (Windows). Usage:
#
#   ./clean_build.sh [options]
#
# Options:
#   --preset NAME   CMake configure preset (default: linux-default).
#                   linux-portable bundles XRT/XDNA libs + static FFmpeg.
#   --hrx           Configure with -DFLM_USE_HRX=ON (HRX amdxdna runtime).
#   -j N            Parallel build jobs (default: nproc).
#   --no-retry      Do not retry configure once after a first-pass failure.
#   --keep-build    Do not delete src/build before configuring.
#                   (Default is to delete it: a failed configure leaves a
#                   CMakeCache behind that silently changes dependency
#                   selection on the next run. See clean_build.bat header.)
#   --ironbuild [VENV]
#                   Also build the AIE design sets (kernels) with the IRON
#                   toolchain after flm links: runs
#                   npu_offload/gemm_rtp/build.sh, which builds the five
#                   families from families.json (~20 min, skipping sets that
#                   are already built). VENV is the iron venv dir holding
#                   bin/python (default: ./ironvenv). Example:
#                     ./clean_build.sh --ironbuild ./ironvenv
#   --iron-only NAME
#                   With --ironbuild: build one design family only.
#   --iron-dev NAME With --ironbuild: IRON device family for the design sets
#                   (default: npu2). Forwarded to npu_offload/gemm_rtp/build.sh.
#   --iron-force    With --ironbuild: rebuild design sets even if present.
#   -h, --help      Show this help.
#
# Documented Linux procedure (README.md, docs/linux-getting-started.md) is:
#   cmake --preset linux-default && cmake --build build && sudo cmake --install .
# This script wraps exactly that, plus the clean-tree + retry + report steps
# from the .bat. Without --ironbuild it does NOT build the AIE design sets
# (IRON toolchain, ~20 min) and only reports whether they are present.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$REPO/src"
BUILD_DIR="$REPO/src/build"

PRESET="linux-default"
EXTRA_CMAKE_ARGS=()
JOBS="$(nproc 2>/dev/null || echo 4)"
RETRY=1
WIPE=1
IRONBUILD=0
IRONVENV=""
IRON_ONLY=""
IRON_DEV=""
IRON_FORCE=0
EXTRA_CMAKE_ARGS+=("-DXRT_DIR=/opt/xilinx/xrt/share/cmake/XRT")
EXTRA_CMAKE_ARGS+=("-DPKG_CONFIG_PATH=/opt/xilinx/xrt/share/pkgconfig")

needval() { [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset=*) PRESET="${1#--preset=}"; shift ;;
        --preset) needval "$@"; PRESET="$2"; shift 2 ;;
        --hrx) EXTRA_CMAKE_ARGS+=("-DFLM_USE_HRX=ON"); shift ;;
        -j=*) JOBS="${1#-j=}"; shift ;;
        -j) needval "$@"; JOBS="$2"; shift 2 ;;
        --no-retry) RETRY=0; shift ;;
        --keep-build) WIPE=0; shift ;;
        --ironbuild=*) IRONBUILD=1; IRONVENV="${1#--ironbuild=}"; shift ;;
        # Bare --ironbuild takes an optional venv: a following non-flag arg
        # is the venv path, otherwise the default ./ironvenv is used.
        --ironbuild)
            IRONBUILD=1
            if [[ $# -ge 2 && "$2" != -* ]]; then IRONVENV="$2"; shift 2; else shift; fi ;;
        --iron-only=*) IRON_ONLY="${1#--iron-only=}"; shift ;;
        --iron-only) needval "$@"; IRON_ONLY="$2"; shift 2 ;;
        --iron-dev=*) IRON_DEV="${1#--iron-dev=}"; shift ;;
        --iron-dev) needval "$@"; IRON_DEV="$2"; shift 2 ;;
        --iron-force) IRON_FORCE=1; shift ;;
        -h|--help)
            awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "ERROR: unknown argument: $1 (try --help)" >&2; exit 1 ;;
    esac
done
if [[ "$IRONBUILD" -eq 0 && ( -n "$IRON_ONLY" || -n "$IRON_DEV" || "$IRON_FORCE" -eq 1 ) ]]; then
    echo "WARNING: --iron-only/--iron-dev/--iron-force have no effect without --ironbuild." >&2
fi

# ---------------------------------------------------------------- prerequisites
die() { echo "ERROR: $*" >&2; exit 1; }

command -v cmake >/dev/null 2>&1 || die "cmake is not on PATH. Install it (e.g. 'sudo apt install cmake')."
command -v ninja >/dev/null 2>&1 || die "ninja is not on PATH. Install it (e.g. 'sudo apt install ninja-build')."
command -v g++ >/dev/null 2>&1 || echo "WARNING: g++ is not on PATH; configure will fail without a C++20 compiler." >&2
command -v pkg-config >/dev/null 2>&1 || echo "WARNING: pkg-config is not on PATH; XRT/FFmpeg/FFTW detection needs it." >&2

if [[ ! -f "$SRC_DIR/CMakePresets.json" ]]; then
    die "no CMake presets at $SRC_DIR/CMakePresets.json"
fi
if [[ ! -f "$SRC_DIR/CMakeLists.txt" ]]; then
    die "no CMakeLists at $SRC_DIR/CMakeLists.txt"
fi

# XRT ships xrt.pc under lib64/pkgconfig, which pkg-config does not search,
# so the configure warns "Package 'xrt' not found" and falls back to the
# manual /opt/xilinx/xrt paths (same libraries -- cosmetic, but the warning
# sends people looking for a missing XRT). Pointing pkg-config at the .pc
# uses the canonical detection instead.
if command -v pkg-config >/dev/null 2>&1 && ! pkg-config --exists xrt 2>/dev/null; then
    for pc in /opt/xilinx/xrt/lib64/pkgconfig /opt/xilinx/xrt/lib/pkgconfig; do
        if [[ -f "$pc/xrt.pc" ]]; then
            export PKG_CONFIG_PATH="$pc${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
            break
        fi
    done
fi

# ---------------------------------------------------------------- configure
if [[ "$WIPE" -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "Removing the existing build tree (a stale CMakeCache silently changes dependency selection)."
    rm -rf "$BUILD_DIR"
fi

configure() {
    echo
    echo "=== configure ($1) [preset: $PRESET] ==="
    # Presets live in src/, so configure from there; binaryDir comes from the preset.
    (cd "$SRC_DIR" && cmake --preset "$PRESET" "${EXTRA_CMAKE_ARGS[@]}")
}

if ! configure "pass 1"; then
    if [[ "$RETRY" -eq 1 ]]; then
        echo
        echo "Pass 1 failed. Retrying once with a wiped build tree"
        echo "(first-pass FetchContent/network failures leave a poisoned cache behind)."
        echo
        rm -rf "$BUILD_DIR"
        echo "=== configure (pass 2) ==="
        (cd "$SRC_DIR" && cmake --preset "$PRESET" "${EXTRA_CMAKE_ARGS[@]}") \
            || die "configure failed twice; the output above is the real diagnostic."
    else
        die "configure failed (--no-retry: not retrying)."
    fi
fi

# ---------------------------------------------------------------- build
echo
echo "=== build (jobs: $JOBS) ==="
cmake --build "$BUILD_DIR" --target flm -j "$JOBS" \
    || die "build failed. The output above is the real diagnostic."

if [[ ! -x "$BUILD_DIR/flm" ]]; then
    die "the build reported success but there is no executable at $BUILD_DIR/flm."
fi

# ---------------------------------------------------------------- report
echo
SIZE="$(du -h "$BUILD_DIR/flm" | cut -f1)"
echo "Built $BUILD_DIR/flm ($SIZE)"
echo
echo "Run it BY FULL PATH the first time:"
echo "    $BUILD_DIR/flm --version"
INSTALLED="$(command -v flm || true)"
if [[ -n "$INSTALLED" && "$INSTALLED" != "$BUILD_DIR/flm" ]]; then
    echo "Note: bare \`flm\` resolves to $INSTALLED, not the binary just built —"
    echo "use the full path above until you install (e.g. 'sudo cmake --install $BUILD_DIR')."
fi
echo

MISSING=""
for fam in BERT-h384-bfp16 BERT-h384-bf16 BERT-h768-bfp16 BERT-h768-gated-bfp16 BERT-h1024-bfp16; do
    if [[ ! -f "$REPO/src/xclbins/$fam/gemm_rtp/design.json" ]]; then
        MISSING="$MISSING $fam"
    fi
done
if [[ "$IRONBUILD" -eq 1 ]]; then
    echo "=== design sets [IRON] ==="
    IRON_ARGS=()
    if [[ -n "$IRONVENV" ]]; then IRON_ARGS+=(--ironvenv "$IRONVENV"); fi
    if [[ -n "$IRON_ONLY" ]]; then IRON_ARGS+=(--only "$IRON_ONLY"); fi
    if [[ -n "$IRON_DEV" ]]; then IRON_ARGS+=(--dev "$IRON_DEV"); fi
    if [[ "$IRON_FORCE" -eq 1 ]]; then IRON_ARGS+=(--force); fi
    "$REPO/npu_offload/gemm_rtp/build.sh" "${IRON_ARGS[@]}" \
        || die "design-set build failed. The output above is the real diagnostic."
    echo
    echo "flm and all requested AIE design sets are built."
elif [[ -n "$MISSING" ]]; then
    echo "The AIE design sets are NOT built:$MISSING"
    echo "An open_npue model will refuse to load until they are. Either re-run"
    echo "with --ironbuild to build them now (IRON toolchain, ~20 min):"
    echo "    ./clean_build.sh --ironbuild ./ironvenv"
    echo "or build them directly per npu_offload/gemm_rtp/README.md:"
    echo "    npu_offload/gemm_rtp/build.sh --ironvenv ./ironvenv"
else
    echo "All five AIE design sets are present."
fi
