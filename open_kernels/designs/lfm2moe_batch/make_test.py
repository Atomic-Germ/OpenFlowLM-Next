r"""Test vectors for mb4: a pool of EXPERTS LFM2.5-24B-A2B-shaped experts (q4_1 chunks, the fl_q4 pool law of
recipes/lfm2moe.py: [gate band b | up band b] x 24 then the down bands), eight distinct token activations per slot, and the fp64 reference
y[slot] = down(silu(gate x) * (up x)) from the SAME blocks the kernel streams.

    python make_test.py --slots 16 [--experts 16] [--seed S] [--runs N] []

Writes pool_e<E>.bin, x_s<S>.bin, ref_s<S>.bin, refh_s<S>.bin and run_s<S>.cfg for harness/run_kernel.
Build the matching kernel with MBL_SLOTS=<S> MBL_EXPERTS=<E>.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from q4_1_pack import CH, dequant_q4_1, pack_q4_1_pool, random_q4_1_blocks  # noqa: E402
sys.path.insert(0, str(HERE))
from layout import FF, HID, NT, x_to_dev  # noqa: E402


def silu(x):
    return x / (1 + np.exp(-x))


def bands(blocks, k):
    """pool-order bytes of a [n, k] Q4_0 matrix as std_perm 64-row bands [n/64][band bytes]."""
    p = pack_q4_1_pool(blocks, rs=2)
    return p.reshape(blocks.shape[0] // 64, -1)


def pack_experts(gates, ups, downs) -> np.ndarray:
    E = len(gates)
    nb_up, nb_dn = FF // 64, HID // 64
    bb_h = 2 * (HID // 256) * CH
    bb_f = 2 * (FF // 256) * CH
    exp_ug = 2 * nb_up * bb_h
    up_bytes = nb_dn * bb_f
    assert up_bytes == nb_up * bb_h
    pool = np.zeros(E * (exp_ug + up_bytes), np.uint8)
    for e in range(E):
        g, u = bands(gates[e], HID), bands(ups[e], HID)
        assert g.shape == (nb_up, bb_h), g.shape
        o = e * exp_ug
        for b in range(nb_up):
            pool[o + (2 * b) * bb_h:o + (2 * b + 1) * bb_h] = g[b]
            pool[o + (2 * b + 1) * bb_h:o + (2 * b + 2) * bb_h] = u[b]
        d = bands(downs[e], FF)
        assert d.shape == (nb_dn, bb_f), d.shape
        pool[E * exp_ug + e * up_bytes:E * exp_ug + (e + 1) * up_bytes] = d.reshape(-1)
    return pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    S = a.slots
    E = a.experts or S
    if S % 8 or S > E:
        print(f"REFUSE: slots={S} must be a multiple of 8 and <= experts={E}")
        return 1
    rng = np.random.default_rng(a.seed)
    print(f"packing {E} experts (gate / up [{FF}, {HID}], down [{HID}, {FF}]) ...")
    gates = [random_q4_1_blocks(FF, HID, rng) for _ in range(E)]
    ups = [random_q4_1_blocks(FF, HID, rng) for _ in range(E)]
    downs = [random_q4_1_blocks(HID, FF, rng, scale=0.05) for _ in range(E)]
    pool = pack_experts(gates, ups, downs)
    xs = rng.standard_normal((S, NT, HID)).astype(np.float32).astype(bfloat16)
    x_dev = x_to_dev(xs)
    print("fp64 reference per slot from the same blocks (bf16 scales, bf16 h) ...")
    ref = np.zeros((S, NT, HID), np.float32)
    ref_h = np.zeros((S, NT, FF), bfloat16)
    for s in range(S):
        wg = dequant_q4_1(gates[s], scale_dtype=bfloat16).astype(np.float64)
        wu = dequant_q4_1(ups[s], scale_dtype=bfloat16).astype(np.float64)
        wd = dequant_q4_1(downs[s], scale_dtype=bfloat16).astype(np.float64)
        xt = xs[s].astype(np.float64)
        h = (silu(xt @ wg.T) * (xt @ wu.T)).astype(np.float32).astype(bfloat16)
        ref_h[s] = h
        ref[s] = (h.astype(np.float64) @ wd.T).astype(np.float32)
    tag = f"s{S}"
    (HERE / f"pool_e{E}.bin").write_bytes(pool.tobytes())
    (HERE / f"x_{tag}.bin").write_bytes(x_dev.tobytes())
    (HERE / f"ref_{tag}.bin").write_bytes(ref.tobytes())
    (HERE / f"refh_{tag}.bin").write_bytes(ref_h.tobytes())
    hb, yb = S * FF * NT * 2, S * HID * NT * 4
    cfg = ["device", f"xclbin G build_{tag}_e{E}/final.xclbin", f"kernelx k G build_{tag}_e{E}/insts.bin",
           f"buf pool {pool.nbytes} pool_e{E}.bin", f"buf x {x_dev.nbytes} x_{tag}.bin", f"buf h {hb}", f"buf y {yb}"]
    cfg += ["run k pool x h y"] * a.runs
    cfg += [f"dump y y_{tag}.bin {yb}", f"dump h h_{tag}.bin {hb}", ""]
    (HERE / f"run_{tag}.cfg").write_text("\n".join(cfg))
    print(f"{tag}: pool={pool.nbytes} B x={x_dev.nbytes} B y={yb} B absmax={np.abs(ref).max():.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
