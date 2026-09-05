#!/bin/bash
# Build a RAW .mlir design (not @iron.jit) into xclbin + insts.bin [+ insts.elf].
# Raw MLIR is the route for designs that need dialect features IRON's Python
# surface doesn't expose -- packet flows to a tile's TileControl port, explicit
# BD register pokes, control packets.
#
#   source ~/ironenv142/bin/activate
#   source ~/mlir-aie/utils/env_setup.sh
#   export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib
#   bash build_mlir.sh designs/expert_fetch/ctrlpkt_shim_bd.mlir [outdir]
set -euo pipefail
SRC="$1"
OUT="${2:-$(dirname "$SRC")/build}"
mkdir -p "$OUT"
cd "$OUT"
aiecc --get-xclbin --get-npu-insts \
      --alloc-scheme=basic-sequential \
      --xclbin-name=final.xclbin --npu-insts-name=insts.bin \
      "$(cd "$(dirname "$OLDPWD/$SRC")" && pwd)/$(basename "$SRC")" 2>&1 | tail -20
ls -la final.xclbin insts.bin 2>/dev/null
echo "BUILD_MLIR_OK -> $OUT"
