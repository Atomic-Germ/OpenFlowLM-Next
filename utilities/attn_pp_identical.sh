#!/bin/bash
# Every family that does NOT set ATTN_BLOCK_ONLY must compile attn.h byte for byte
# (OPEN-ATTN-CONTEXT: a family joins the path by measurement, and everything else
# compiles what it compiled). Prove it at the source level: preprocess each attention
# translation unit at a family's own flags with the pre-change header set and with the
# current one, and diff. A byte-identical preprocessed TU is a byte-identical object,
# which is the same evidence a --check re-export gives for one tenth of the time.
#
# Run from WSL, from the repo root, with the reference sources extracted first:
#
#   mkdir -p ppcheck_old
#   for f in attn.h attn_stepb.cc; do   # every file the change touched
#     git show <ref>:open_kernels/designs/attn/$f > ppcheck_old/$f
#   done
#   wsl -d Ubuntu-24.04 -- bash -lc 'source ~/ironenv142/bin/activate &&
#     bash <repo>/utilities/attn_pp_identical.sh'
#
# On a Windows checkout with core.autocrlf the working copy of this file has CRLF line
# endings and bash under WSL dies on line 1; strip the CRs (or check it out with LF).
#
# Add a family by adding its ATTN_* flags below -- they are recipes/attnknobs.py's
# geometry for that spec, which `python -c "from recipes import dense; ..."` prints.
set -e
W=/mnt/c/code/openflowlm-next/.claude/worktrees/agent-a7a473d4917f6cad9
A=$W/open_kernels/designs/attn
OUT=/tmp/ppcheck
rm -rf $OUT; mkdir -p $OUT/old $OUT/new
cp $A/*.cc $A/*.h $OUT/new/
cp $A/*.cc $A/*.h $OUT/old/
cp $W/ppcheck_old/* $OUT/old/   # the reference set: every file the change touched
rm -f $OUT/old/attn_stepb_new.cc

eval "$(python - <<'PY'
from aie.utils import config
print(f'PEANO={config.peano_install_dir()}')
print(f'HDR={config.cxx_header_path()}')
PY
)"
ARCH=aie2p
CLANG="$PEANO/bin/clang++"
BASE="-I$HDR -I$HDR/aie_kernels -I$HDR/aie_kernels/$ARCH -I$W/open_kernels/include -D__AIE_API_AIE_ADF_HPP__ --target=$ARCH-none-unknown-elf -std=c++20 -O2 -DNDEBUG -Wno-macro-redefined"

# Gemma3-4B: head dim 256, NO gate, 4 cores x 2 heads, RB 2 -- the family that keeps the
# single-row kernel AT a blocked RB, so it exercises every branch this change touched.
G3="-DATTN_NH=8 -DATTN_KVH=4 -DATTN_HD=256 -DATTN_ROT=128 -DATTN_GATE=0 -DATTN_VEXP=1 -DATTN_NHL=2 -DATTN_RB=2 -DATTN_QKNORM=1"
# Qwen3-4B: head dim 128, 4 cores x 8 heads, RB 4 -- a family NOT at hd 256.
Q3="-DATTN_NH=32 -DATTN_KVH=8 -DATTN_HD=128 -DATTN_ROT=128 -DATTN_GATE=0 -DATTN_VEXP=1 -DATTN_NHL=8 -DATTN_RB=4 -DATTN_QKNORM=1"
# The 35B as it shipped: hd 256 WITH the gate, RB 1, the single-row path.
Q36="-DATTN_NH=16 -DATTN_KVH=2 -DATTN_HD=256 -DATTN_ROT=64 -DATTN_GATE=1 -DATTN_VEXP=1 -DATTN_NHL=4"

fail=0
for name in "gemma3-4b|$G3" "qwen3-4b|$Q3" "qwen36-35b-rb1|$Q36"; do
  tag=${name%%|*}; flags=${name#*|}
  for tu in attn_meta.cc attn_q.cc attn_k.cc attn_v.cc attn_init.cc attn_fin.cc attn_fin_ng.cc attn_step.cc attn_step_new.cc attn_stepb.cc; do
    [ -f "$OUT/old/$tu" ] || continue
    if [ "$tu" = attn_stepb.cc ]; then case "$flags" in *ATTN_RB=*) ;; *) continue;; esac; fi
    ok=1
    for v in old new; do
      "$CLANG" -E -P $BASE -I$OUT/$v $flags "$OUT/$v/$tu" -o "$OUT/$tag.$tu.$v.i" || ok=0
      # An empty macro expansion leaves a stray space (`int e )`): compare TOKENS, not
      # bytes -- whitespace collapsed and dropped beside punctuation. Anything that would
      # change the compiled code changes a token.
      [ $ok = 1 ] && sed -i "s|$OUT/$v|DIR|g" "$OUT/$tag.$tu.$v.i" &&         tr -s '[:space:]' ' ' < "$OUT/$tag.$tu.$v.i" | sed 's/ *\([][(){},;:<>=*&+-]\) *//g' > "$OUT/$tag.$tu.$v.t"
    done
    [ $ok = 1 ] || { echo "PP FAILED $tag $tu"; fail=1; continue; }
    if cmp -s "$OUT/$tag.$tu.old.t" "$OUT/$tag.$tu.new.t"; then
      echo "same  $tag  $tu"
    else
      echo "DIFF  $tag  $tu"; diff <(tr ";" "
" < "$OUT/$tag.$tu.old.t") <(tr ";" "
" < "$OUT/$tag.$tu.new.t") | head -20; fail=1
    fi
  done
done
echo "PPCHECK_FAIL=$fail"
