#!/usr/bin/env bash
# One-command fair parity for the merged `lax` one-submit decode (goal mug03nrk-zekgbf).
# Usage: fair_parity.sh DEC_DIR LAX_L_DIR LAX_A_DIR MODEL_DIR POOL_DIR [TOKENS]
# The reference adopts the NPU's top-8 (make_decode's DEFAULT protocol); the routing words come from
# the plain-run pass, which is numerically identical to the one-submit loop (byte-identical logits).
set -euo pipefail
DEC=$1; LAX_L=$2; LAX_A=$3; MD=$4; POOLS=$5; TOK=${6:-16}
cd "$(dirname "$0")/.."                                   # -> open_kernels/
export OPEN_KERNELS_SPEC=${OPEN_KERNELS_SPEC:-recipes/specs/qwen36-35b-a3b.json}
export MOE_ONDEVICE_ROUTE=1 ONDV_EMIT_SHARED=1 ONDV_PKTDONE_ACQ=1
python model/lax_decode_cfg.py --out "$DEC" --lax-l "$LAX_L" --lax-a "$LAX_A" --per 1 --tokens "$TOK" --dump-res
python model/plain_layer_cfg.py "$DEC/run_lax_p1.cfg" "$DEC/run_plain.cfg" "$LAX_L" "$LAX_A"
python model/route_dump_cfg.py "$DEC/run_plain.cfg" "$DEC/run_route.cfg"
flock ~/.cache/lax-decode/box.lock harness/build/run_kernel "$DEC/run_route.cfg"
python model/make_decode.py --requant --model-dir "$MD" --layers 40 --tokens "$TOK" --out "$DEC" \
       --pool-dir "$POOLS" --reuse-pools
python model/compare_decode.py --tokens "$TOK" --out "$DEC"
