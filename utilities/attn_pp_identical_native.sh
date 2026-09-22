#!/bin/bash
# Native-Windows twin of attn_pp_identical.sh (Git Bash + the ironenv's Peano): preprocess
# every attention TU at each non-35B family's own flags with a reference header set and with
# the working tree's, and compare token streams. Identical tokens = identical objects.
#
#   bash utilities/attn_pp_identical_native.sh <git ref>      # e.g. decode-gap/integrate or main
#
# Every file under open_kernels/designs/attn and open_kernels/include is taken from the ref.
set -e
REF=${1:-main}
W=$(cd "$(dirname "$0")/.." && pwd -W)
PEANO=C:/dev/mlir-aie/ironenv/Lib/site-packages/llvm-aie
HDR=C:/dev/mlir-aie/ironenv/Lib/site-packages/mlir_aie/include
OUT=$W/.claude/tmp/ppcheck
rm -rf "$OUT"; mkdir -p "$OUT/old/attn" "$OUT/old/include" "$OUT/new"
for f in $(git -C "$W" ls-tree --name-only "$REF" open_kernels/designs/attn/); do
  git -C "$W" show "$REF:$f" > "$OUT/old/attn/$(basename "$f")"
done
for f in $(git -C "$W" ls-tree --name-only "$REF" open_kernels/include/); do
  git -C "$W" show "$REF:$f" > "$OUT/old/include/$(basename "$f")"
done
CLANG="$PEANO/bin/clang++.exe"
base() { echo "-I$HDR -I$HDR/aie_kernels -I$HDR/aie_kernels/aie2p -I$1 -D__AIE_API_AIE_ADF_HPP__ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -Wno-macro-redefined"; }
G3="-DATTN_NH=8 -DATTN_KVH=4 -DATTN_HD=256 -DATTN_ROT=128 -DATTN_GATE=0 -DATTN_VEXP=1 -DATTN_NHL=2 -DATTN_RB=2 -DATTN_QKNORM=1"
Q3="-DATTN_NH=32 -DATTN_KVH=8 -DATTN_HD=128 -DATTN_ROT=128 -DATTN_GATE=0 -DATTN_VEXP=1 -DATTN_NHL=8 -DATTN_RB=4 -DATTN_QKNORM=1"
Q35="-DATTN_NH=16 -DATTN_KVH=4 -DATTN_HD=256 -DATTN_ROT=64 -DATTN_GATE=1 -DATTN_VEXP=1 -DATTN_NHL=4"
Q36RB1="-DATTN_NH=16 -DATTN_KVH=2 -DATTN_HD=256 -DATTN_ROT=64 -DATTN_GATE=1 -DATTN_VEXP=1 -DATTN_NHL=4"
fail=0; n=0
for name in "gemma3-4b|$G3" "qwen3-4b|$Q3" "qwen35-9b|$Q35" "qwen36-rb1|$Q36RB1"; do
  tag=${name%%|*}; flags=${name#*|}
  for tu in attn_meta attn_q attn_k attn_v attn_init attn_fin attn_fin_ng attn_step attn_step_new attn_stepb; do
    if [ $tu = attn_stepb ]; then case "$flags" in *ATTN_RB=*) ;; *) continue;; esac; fi
    "$CLANG" -E -P $(base "$OUT/old/include") -I"$OUT/old/attn" $flags "$OUT/old/attn/$tu.cc" -o "$OUT/$tag.$tu.old.i"
    "$CLANG" -E -P $(base "$W/open_kernels/include") -I"$W/open_kernels/designs/attn" $flags "$W/open_kernels/designs/attn/$tu.cc" -o "$OUT/$tag.$tu.new.i"
    n=$((n + 1))
    if cmp -s <(tr -s ' \t\n' '\n' < "$OUT/$tag.$tu.old.i") <(tr -s ' \t\n' '\n' < "$OUT/$tag.$tu.new.i"); then
      echo "identical  $tag  $tu"
    else
      echo "DIFFERENT  $tag  $tu"; fail=1
    fi
  done
done
echo "$n TUs compared against $REF"
exit $fail
