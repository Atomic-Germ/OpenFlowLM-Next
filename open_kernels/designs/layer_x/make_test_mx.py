"""Synthetic whole-layer MoE block gate (mx) at this spec's geometry.

Builds a dense-factored pool with random q4_1 expert weights in the SAME
pool order the pack plan places (interleaved stripes RS=4 + down RS=4),
runs mx.py (routed only) via moe2 patching from a synthetic router record,
and compares the block output against an fp64 reference from the SAME pool
bytes (dequant_pool). Proves moe + gemv_q4_prep_f32 together at (ff, E, topk).

    OPEN_KERNELS_SPEC=.../lfm2moe-24b-a2b.json MX_KIND=full python make_test_mx.py
    run_kernel run_mx.cfg ; python compare_mx.py
Compare tolerance mirrors moe_chain's (cos > 0.9999, maxrel < 5e-3).
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import fixture_paths as FX  # noqa: E402
from layout import (AA_BYTES, AA_HP, AA_RES, AA_ROUT, AA_XM, CA_BYTES, CA_SGW,  # noqa: E402
                    KV_BYTES, POOL_BYTES, POOL_DOWN, PTAB_BYTES, R, SPEC)
from recipes import qwen36moe as Q  # noqa: E402
from q4_1_pack import random_q4_1_blocks, pack_q4_1_pool, dequant_pool  # noqa: E402

C = R.common
E, TOPK = int(SPEC.num_experts), int(SPEC.experts_per_tok)
HID, FF = C.HID, C.FF  # padded ff
S, SPP = C.STRIPE, C.STRIPES_PER_PROJ
OUT_PER = 1024  # f32 per router record: [p E | idx 8 | w 8] padded; only idx/w staged

def silu(x):
    return x / (1 + np.exp(-x))

def main() -> int:
    rng = np.random.default_rng(1)
    pool = np.zeros(POOL_BYTES, np.uint8)
    # Expert weights: per expert, up/gate/down as pool-order bytes from random blocks.
    up_floats = []
    gate_floats = []
    down_floats = []
    for e in range(E):
        # Up/gate: full [FF,HID] matrices packed RS=4 in one go (as the pack does),
        # then sliced stripe-wise and interleaved [up_k|gate_k] like expert_stripes.
        ub_full = pack_q4_1_pool(random_q4_1_blocks(FF, HID, rng), 4)
        gb_full = pack_q4_1_pool(random_q4_1_blocks(FF, HID, rng), 4)
        upf = dequant_pool(ub_full, FF, HID, 4)
        gatef = dequant_pool(gb_full, FF, HID, 4)
        for k in range(SPP):
            u_off = (2 * SPP * e + 2 * k) * S
            pool[u_off:u_off + S] = ub_full[k * S:(k + 1) * S]
            pool[u_off + S:u_off + 2 * S] = gb_full[k * S:(k + 1) * S]
        db = pack_q4_1_pool(random_q4_1_blocks(HID, FF, rng), 4)
        doff = POOL_DOWN + e * C.UP_BYTES
        pool[doff:doff + C.UP_BYTES] = db[:C.UP_BYTES]
        df = dequant_pool(db, HID, FF, 4)
        up_floats.append(upf); gate_floats.append(gatef); down_floats.append(df)
    xm = (rng.standard_normal(HID) * 0.5).astype(np.float32).astype(bfloat16)
    xres = (rng.standard_normal(HID) * 0.5).astype(np.float32)
    # router: choose topk distinct experts deterministically
    idx = [3, 11, 23, 41][:TOPK] if TOPK <= 4 else list(range(TOPK))
    w = np.array([1.0 / TOPK] * TOPK, np.float64)
    # Reference block output
    acc = np.zeros(HID, np.float64)
    xm_f = xm.astype(np.float64)
    for s in range(TOPK):
        e = idx[s]
        u = up_floats[e].astype(np.float64) @ xm_f
        g = gate_floats[e].astype(np.float64) @ xm_f
        h = (silu(g) * u).astype(np.float32).astype(bfloat16).astype(np.float64)
        y = down_floats[e].astype(np.float64) @ h
        acc += w[s] * y
    ref = xres.astype(np.float64) + acc
    # Stage buffers
    consts = np.zeros(CA_BYTES, np.uint8)
    consts[CA_SGW:CA_SGW + 4096] = np.zeros(4096, np.uint8)  # no shared: zero gate
    act = np.zeros(AA_BYTES, np.uint8)
    act[AA_XM:AA_XM + 4096] = np.frombuffer(xm.view(np.uint8).tobytes(), np.uint8)[:4096]
    rec = np.zeros(1024, np.float32)
    rec[E:E + 8] = 0
    rec[E:E + TOPK] = np.array(idx, np.int32).view(np.float32)
    rec[E + 8:E + 8 + TOPK] = w.astype(np.float32)
    act[AA_ROUT:AA_ROUT + 4096] = np.frombuffer(rec.tobytes(), np.uint8)[:4096]
    act[AA_RES:AA_RES + HID * 4] = np.frombuffer(xres.tobytes(), np.uint8)[:HID * 4]
    (HERE / "pool_mx.bin").write_bytes(pool.tobytes())
    (HERE / "consts_mx.bin").write_bytes(consts.tobytes())
    (HERE / "act_mx.bin").write_bytes(act.tobytes())
    (HERE / "xres_mx.bin").write_bytes(xres.tobytes())
    (HERE / "ref_mx.bin").write_bytes(ref.astype(np.float32).tobytes())
    cfg = [
        "device",
        f"moegeom {E} {TOPK} {S} {C.UP_BYTES} {C.DOWN_PER_CORE * C.DOWN_BAND} {POOL_DOWN} {POOL_DOWN + E * C.UP_BYTES} {POOL_DOWN + E * C.UP_BYTES} {POOL_DOWN + E * C.UP_BYTES}",
        "xclbin X build_mx_full_lfm/final.xclbin",
        "kernelx mx X build_mx_full_lfm/insts.bin",
        f"buf pool {POOL_BYTES} pool_mx.bin",
        f"buf xres {HID*4} xres_mx.bin",
        f"buf consts {CA_BYTES} consts_mx.bin",
        f"buf state {KV_BYTES}",
        f"buf act {AA_BYTES} act_mx.bin",
        f"buf ptab {PTAB_BYTES}",
        f"moeroute2 mx act {AA_ROUT + E * 4}",
        "run mx pool xres consts state act ptab",
        f"dump xres y_mx.bin {HID*4}",
        "",
    ]
    (HERE / "run_mx.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"mx: E={E} topk={TOPK} ff={FF} idx={idx} ref max={np.abs(ref).max():.4g}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
