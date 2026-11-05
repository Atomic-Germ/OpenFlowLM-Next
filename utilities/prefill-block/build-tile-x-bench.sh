#!/usr/bin/env bash
# Build the isolated tile_x benchmark (tile-x-bench.cpp).
#
# It links the real `host::tile_x` from block_host.cpp rather than a copy, so the "before"
# column is the function the engine actually calls -- compiled with the same
# -ffp-contract=off the engine's build gives it (that flag is not cosmetic: it stops the
# compiler fusing a multiply-add, which would change the host arithmetic).
#
#   ./utilities/prefill-block/build-tile-x-bench.sh [outdir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/Models/qwen38-27b/prefill-perf/tile-x}"

# Same resolution as utilities/prefill-block/build-cli.sh: the runtime package ships the
# library but not the dev headers, and it does not install an unversioned .so.
XRT="${XRT_DIR:-${XRT_ROOT:-${XILINX_XRT:-/opt/xilinx/xrt}}}"
XRT_INC=""
for cand in "$XRT" /opt/xilinx/xrt \
            /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt; do
    if [ -f "$cand/include/xrt/xrt_kernel.h" ]; then XRT_INC="$cand/include"; break; fi
done
if [ -z "$XRT_INC" ]; then
    echo "build-tile-x-bench: no xrt_kernel.h under any known XRT prefix (set XRT_DIR)" >&2
    exit 1
fi
XRT_LIB="$XRT/lib"
[ -d "$XRT_LIB" ] || XRT_LIB="$(dirname "$XRT_INC")/lib"
if [ -f "$XRT_LIB/libxrt_coreutil.so" ]; then LIBXRT="$XRT_LIB/libxrt_coreutil.so";
else LIBXRT="$XRT_LIB/libxrt_coreutil.so.2"; fi

CXX="${CXX:-g++}"
mkdir -p "$OUT"

"$CXX" -O3 -DNDEBUG -std=c++20 -fopenmp -mavx2 -mfma -ffp-contract=off \
       -I"$ROOT/src" -I"$ROOT/src/include" -I"$XRT_INC" \
       "$ROOT/utilities/prefill-block/tile-x-bench.cpp" "$ROOT/src/open_qwen36/block_host.cpp" \
       -L"$XRT_LIB" -Wl,-rpath,"$XRT_LIB" "$LIBXRT" -o "$OUT/tile_x_bench"
echo "build-tile-x-bench: $OUT/tile_x_bench"
