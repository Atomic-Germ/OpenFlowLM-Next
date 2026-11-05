#!/usr/bin/env bash
# Build the standalone open_qwen36_cli harness by hand.
#
# Why this exists: `src/open_qwen36/cli.cpp` is DELIBERATELY excluded from
# src/CMakeLists.txt (it has its own main(), as open_whisper's and open_diffusion's do),
# so `oflm` does not carry it and there is no CMake target for it. The Phase 22-26
# harnesses all drive this binary, and it has to be rebuilt on the host whenever
# core.cpp / cli.cpp / block_host.cpp change.
#
# The command below is the one CMake generated for the same target in
# Models/qwen38-27b/prefill-perf/switch-v1/build (CMakeFiles/open_qwen36_cli.dir/),
# with the container's /code prefix replaced by this tree. Reproducing it here keeps
# the harness buildable from a clean checkout instead of only inside an old build dir.
#
#   ./utilities/prefill-block/build-cli.sh [outdir]
#
# Default outdir: Models/qwen38-27b/prefill-perf/block-v1/build
#
# NOT switch-v1/build: that tree was configured inside a container (its CMakeCache
# records /code/...) and is root-owned here, so cmake cannot re-run in it and an
# ordinary user cannot write to it at all.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/Models/qwen38-27b/prefill-perf/block-v1/build}"
# XRT_ROOT is what utilities/env.sh and the XRT packages themselves set; XILINX_XRT is
# the older name. Both are honoured so this works with env.sh sourced or not.
XRT="${XRT_DIR:-${XRT_ROOT:-${XILINX_XRT:-/opt/xilinx/xrt}}}"

# HEADERS and LIBS are resolved SEPARATELY on purpose. The installed runtime package
# ships /opt/xilinx/xrt/lib but only a stub include/ (vaccel.h) -- the dev headers come
# from a separate package. A from-source xdna-driver build has both, so fall back to it
# instead of failing, and say which was used.
XRT_INC=""
for cand in "$XRT" /opt/xilinx/xrt \
            /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt; do
    if [ -f "$cand/include/xrt/xrt_kernel.h" ]; then XRT_INC="$cand/include"; break; fi
done
if [ -z "$XRT_INC" ]; then
    echo "build-cli: no xrt_kernel.h under any known XRT prefix (set XRT_DIR)" >&2
    exit 1
fi
XRT_LIB="$XRT/lib"
if [ ! -d "$XRT_LIB" ]; then XRT_LIB="$(dirname "$XRT_INC")/lib"; fi
if [ ! -f "$XRT_LIB/libxrt_coreutil.so" ] && [ ! -f "$XRT_LIB/libxrt_coreutil.so.2" ]; then
    echo "build-cli: no libxrt_coreutil in $XRT_LIB" >&2
    exit 1
fi
echo "build-cli: headers $XRT_INC"
echo "build-cli: libs    $XRT_LIB"

SRC="$ROOT/src/open_qwen36"
mkdir -p "$OUT"
OBJ="$OUT/obj26"
mkdir -p "$OBJ"

CXX="${CXX:-g++}"
# -O3 -DNDEBUG and the three include dirs are the generated ones. OpenMP is used
# directly by core.cpp / block_host.cpp, so -fopenmp and the gomp link are required.
FLAGS=(-O3 -DNDEBUG -std=c++20 -fopenmp
       -I"$SRC/.." -I"$SRC/../include" -I"$ROOT/open_kernels/harness"
       -I"$XRT_INC")
# src/CMakeLists.txt:666 gives block_host.cpp its own compile options, and the third one
# is not cosmetic: -ffp-contract=off stops the compiler fusing a multiply-add into an FMA,
# which would change the host arithmetic the whole prefill is validated against. The AVX2
# is what its intrinsics need. Reproduced per-unit rather than globally so the other units
# keep the same code generation CMake gave them.
declare -A UNIT_FLAGS=(
    [block_host]="-mavx2 -mfma -ffp-contract=off"
)

UNITS=(cli core block_host pools manifest q4nx_file gguf_file)

echo "build-cli: $CXX -> $OUT/open_qwen36_cli"
for u in "${UNITS[@]}"; do
    printf '  %-12s ' "$u"
    # shellcheck disable=SC2086
    "$CXX" "${FLAGS[@]}" ${UNIT_FLAGS[$u]:-} -c "$SRC/$u.cpp" -o "$OBJ/$u.o"
    echo ok
done

# The unversioned .so is not installed by the runtime package, so prefer it and fall
# back to the versioned soname by full path -- `-lxrt_coreutil` cannot resolve .so.2.
if [ -f "$XRT_LIB/libxrt_coreutil.so" ]; then LIBXRT="-lxrt_coreutil"; else LIBXRT="$XRT_LIB/libxrt_coreutil.so.2"; fi
"$CXX" -O3 -DNDEBUG -fopenmp "${OBJ}"/*.o -o "$OUT/open_qwen36_cli" \
       -L"$XRT_LIB" -Wl,-rpath,"$XRT_LIB" $LIBXRT
echo "build-cli: $(stat -c%s "$OUT/open_qwen36_cli") bytes -> $OUT/open_qwen36_cli"
