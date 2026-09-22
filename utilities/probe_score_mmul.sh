#!/bin/bash
# Price the attention score phase written as an aie::mmul block product against the
# per-(row, head) reduce_add form attn_rowb_impl uses today. Compile only: no design
# build, no NPU, no box lock, about three seconds. Program memory on the attention core
# is what decides whether that rewrite is possible, so this is the number to get first.
#
#   wsl -d Ubuntu-24.04 -- bash /mnt/c/<repo>/utilities/probe_score_mmul.sh
#
# Set W to the repo as WSL sees it. Both functions live in probe_score_mmul.cc beside
# this script; neither is wired into a design, and neither is under designs/attn, whose
# *.cc glob feeds the build key.
set -x
W=/mnt/c/code/openflowlm-next
S=$W/utilities
source ~/ironenv142/bin/activate
eval "$(python - <<'PY'
from aie.utils import config
print(f'PEANO={config.peano_install_dir()}')
print(f'HDR={config.cxx_header_path()}')
PY
)"
ARCH=aie2p
FLAGS="-DATTN_NH=16 -DATTN_KVH=2 -DATTN_HD=256 -DATTN_ROT=64 -DATTN_GATE=1 -DATTN_VEXP=1 -DATTN_NHL=4 -DATTN_RB=4 -DATTN_BLOCK_ONLY=1"
BASE="-I$HDR -I$HDR/aie_kernels -I$HDR/aie_kernels/$ARCH -I$W/open_kernels/include -I$W/open_kernels/designs/attn -D__AIE_API_AIE_ADF_HPP__ --target=$ARCH-none-unknown-elf -std=c++20 -O2 -DNDEBUG -Wno-macro-redefined"
mkdir -p /tmp/mmulprobe
"$PEANO/bin/clang++" -c $BASE $FLAGS "$S/probe_score_mmul.cc" -o /tmp/mmulprobe/probe.o
echo "COMPILE_EXIT=$?"
"$PEANO/bin/llvm-nm" --print-size --size-sort /tmp/mmulprobe/probe.o | tail -20
echo "--- the current block kernel at the same flags, for comparison ---"
"$PEANO/bin/clang++" -c $BASE $FLAGS "$W/open_kernels/designs/attn/attn_stepb.cc" -o /tmp/mmulprobe/stepb_rb4.o
"$PEANO/bin/llvm-nm" --print-size --size-sort /tmp/mmulprobe/stepb_rb4.o | tail -20
