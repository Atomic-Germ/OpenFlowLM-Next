r"""Test vectors for moe_batch: a pool of EXPERTS routed experts packed by the
packer's own stripe and down laws (recipes/pack.py), eight distinct token
activations per slot, and the fp64 reference y[slot] = down(silu(gate x) * (up x))
from the SAME pool bytes the kernel streams (../gemv_q4/make_test.py's discipline).

    python make_test.py --slots 16 [--experts 16] [--seed S] [--runs N]

Writes pool_e<E>.bin, x_s<S>.bin (the kernel's A tiles, layout.py), ref_s<S>.bin
(y, [slots, 8 tokens, HID] f32), refh_s<S>.bin (h, [slots, 8, FF] bf16) and
run_s<S>.cfg for the harness; compare.py un-interleaves the kernel's y and h.
Build the matching kernel with MB_SLOTS=<S> MB_EXPERTS=<E>.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))                       # open_kernels/
from q4_1_pack import CH, dequant_q4_1, pack_q4_1_pool, random_q4_1_blocks  # noqa: E402
from recipes.pack import down_perm, stripe_transpose  # noqa: E402
sys.path.insert(0, str(HERE))
from layout import x_to_dev  # noqa: E402

HID, FF, NT = 2048, 512, 8
STRIPE = 128 * HID * 5 // 8
EXP_UG = 2 * (FF // 128) * STRIPE
DOWN_BYTES = HID * FF * 5 // 8


def pack_experts(ups, gates, downs) -> np.ndarray:
    """The pool as recipes/pack.py's expert_stripes + expert_down ops write it: per expert
    the up / gate stripes interleaved (each transposed inside its 128 rows), then, from
    POOL_DOWN, each expert's down in the RS=4 down law."""
    E = len(ups)
    pool = np.zeros(E * (EXP_UG + DOWN_BYTES), np.uint8)
    ncol = HID // 256
    tp = stripe_transpose(HID)
    nchs = STRIPE // CH
    for e in range(E):
        up = pack_q4_1_pool(ups[e], rs=1).reshape(-1, CH)       # file order: 32-row blocks, k-tile minor
        gt = pack_q4_1_pool(gates[e], rs=1).reshape(-1, CH)
        for k in range(FF // 128):
            c = k * nchs
            d = e * EXP_UG + 2 * k * STRIPE
            pool[d:d + STRIPE] = up[c:c + nchs][tp].reshape(-1)
            pool[d + STRIPE:d + 2 * STRIPE] = gt[c:c + nchs][tp].reshape(-1)
    base = E * EXP_UG
    ndn = DOWN_BYTES // CH
    dp = down_perm(ndn)
    for e in range(E):
        dn = pack_q4_1_pool(downs[e], rs=1).reshape(-1, CH)
        pool[base + e * DOWN_BYTES:base + (e + 1) * DOWN_BYTES] = dn[dp].reshape(-1)
    assert ncol == HID // 256
    return pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=16, help="MB_SLOTS the kernel was built with (a multiple of 8)")
    ap.add_argument("--experts", type=int, default=0, help="experts in the pool (MB_EXPERTS); default = slots")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    S = a.slots
    E = a.experts or S
    if S % 8 or S > E:
        print(f"REFUSE: slots={S} must be a multiple of 8 and <= experts={E}")
        return 1
    rng = np.random.default_rng(a.seed)

    print(f"packing {E} experts (up / gate [{FF}, {HID}], down [{HID}, {FF}]) ...")
    ups = [random_q4_1_blocks(FF, HID, rng) for _ in range(E)]
    gates = [random_q4_1_blocks(FF, HID, rng) for _ in range(E)]
    downs = [random_q4_1_blocks(HID, FF, rng, scale=0.05) for _ in range(E)]
    pool = pack_experts(ups, gates, downs)

    print(f"drawing {S} x {NT} distinct token activations ...")
    xs = rng.standard_normal((S, NT, HID)).astype(np.float32).astype(bfloat16)     # [slot, token, K]
    assert len({xs[s, j].tobytes() for s in range(S) for j in range(NT)}) == S * NT
    x_dev = x_to_dev(xs)                                                            # the A tiles

    print("fp64 reference per slot from the same bytes (bf16 scales, bf16 h) ...")
    ref = np.zeros((S, NT, HID), np.float32)
    ref_h = np.zeros((S, NT, FF), bfloat16)
    for s in range(S):
        wu = dequant_q4_1(ups[s], scale_dtype=bfloat16).astype(np.float64)
        wg = dequant_q4_1(gates[s], scale_dtype=bfloat16).astype(np.float64)
        wd = dequant_q4_1(downs[s], scale_dtype=bfloat16).astype(np.float64)
        xt = xs[s].astype(np.float64)                                               # [token, K]
        u, g = xt @ wu.T, xt @ wg.T                                                 # [token, FF]
        h = (g / (1 + np.exp(-g)) * u).astype(np.float32).astype(bfloat16)
        ref_h[s] = h
        ref[s] = (h.astype(np.float64) @ wd.T).astype(np.float32)                   # [token, HID]

    tag = f"s{S}"
    (HERE / f"pool_e{E}.bin").write_bytes(pool.tobytes())
    (HERE / f"x_{tag}.bin").write_bytes(x_dev.tobytes())
    (HERE / f"ref_{tag}.bin").write_bytes(ref.tobytes())
    (HERE / f"refh_{tag}.bin").write_bytes(ref_h.tobytes())
    hb, yb = S * FF * NT * 2, S * HID * NT * 4
    cfg = ["device",
           f"xclbin G build_{tag}_e{E}/final.xclbin",
           f"kernelx k G build_{tag}_e{E}/insts.bin",
           f"buf pool {pool.nbytes} pool_e{E}.bin",
           f"buf x {x_dev.nbytes} x_{tag}.bin",
           f"buf h {hb}",
           f"buf y {yb}"]
    cfg += ["run k pool x h y"] * a.runs
    cfg += [f"dump y y_{tag}.bin {yb}", f"dump h h_{tag}.bin {hb}", ""]
    (HERE / f"run_{tag}.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"{tag}: pool={pool.nbytes} B ({E} experts, POOL_DOWN={E * EXP_UG}) x={x_dev.nbytes} B y={yb} B "
          f"absmax={np.abs(ref).max():.4g}")
    print(f"build: MB_SLOTS={S} MB_EXPERTS={E} python ../../build_design.py moe_batch.py build_{tag}_e{E}")
    print(f"run:   ..\\..\\harness\\out\\run_kernel.exe run_{tag}.cfg && python compare.py {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
