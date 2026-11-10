"""Synthetic whole-layer FULL-attention gate (ax0) at this spec's geometry.

Builds a dense-factored pool with random q4_1 q/k/v/o weights in the SAME
pool order the pack plan places (std_perm), random bf16 consts, runs ax.py
part 0 (through the router) at position 0 via the run_kernel harness, and
dumps the router record. At position 0 the RoPE is the identity and the
attention softmax has one row, so the fp64 reference is exact and simple:
og = v. The reference covers ln, q/k/v, qk-norm, o, residual, post-ln,
router logits, sigmoid + bias selection, renormalisation.

    OPEN_KERNELS_SPEC=.../lfm2moe-8b-a1b.json python make_test_ax_syn.py
    run_kernel run_ax_syn.cfg ; python compare_ax_syn.py
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
from layout import (AA_BYTES, AA_ROUT, AA_XN, CA_BYTES, CA_BIAS, CA_LNW, CA_META, CA_POSTLN, CA_RW,  # noqa: E402
                    KV_BYTES, POOL_BYTES, POOL_K, POOL_O, POOL_Q, POOL_V, PTAB_BYTES, ptab)
from recipes import qwen36moe as Q  # noqa: E402
from q4_1_pack import random_q4_1_blocks, pack_q4_1_pool, dequant_pool  # noqa: E402

from recipes.load import load_spec  # noqa: E402
SPEC = load_spec(os.environ["OPEN_KERNELS_SPEC"])
HID, HD = SPEC.hidden, SPEC.head_dim
NH, KVH = SPEC.num_heads, SPEC.num_kv_heads
E, TOPK = SPEC.num_experts, SPEC.experts_per_tok
QW, KVW = NH * HD, KVH * HD
EPS = SPEC.norm_eps


def rms(x, w):
    v = (x.astype(np.float64) ** 2).mean()
    return (x.astype(np.float64) * (w.astype(np.float64) / np.sqrt(v + EPS))).astype(np.float32)


def silu(x):
    return x / (1 + np.exp(-x))


def main() -> int:
    rng = np.random.default_rng(2)
    pool = np.zeros(POOL_BYTES, np.uint8)
    # Projection weights: [out, HID] matrices packed RS=4 whole, then placed at the
    # role's pool offset as std_perm would (32-row-block-major == pack order).
    mats = {}
    for name, rows, off in (("q", QW, POOL_Q), ("k", KVW, POOL_K), ("v", KVW, POOL_V), ("o", HID, POOL_O)):
        b = pack_q4_1_pool(random_q4_1_blocks(rows, HID, rng), 4)
        mats[name] = dequant_pool(b, rows, HID, 4)
        pool[off:off + len(b)] = b[:]
    xres = (rng.standard_normal(HID) * 0.5).astype(np.float32)
    lnw = (rng.standard_normal(HID) * 0.05 + 1).astype(np.float32).astype(bfloat16)
    postln = (rng.standard_normal(HID) * 0.05 + 1).astype(np.float32).astype(bfloat16)
    qn = (rng.standard_normal(HD) * 0.05 + 1).astype(np.float32).astype(bfloat16)
    kn = (rng.standard_normal(HD) * 0.05 + 1).astype(np.float32).astype(bfloat16)
    rw = (rng.standard_normal(HID * E) * 0.3).astype(np.float32).astype(bfloat16)
    bias = (rng.standard_normal(E) * 0.1).astype(np.float32)

    # ---- fp64 reference
    xn = rms(xres, lnw.astype(np.float32))
    q = (mats["q"].astype(np.float64) @ xn.astype(np.float64)).astype(np.float32)
    k = (mats["k"].astype(np.float64) @ xn.astype(np.float64)).astype(np.float32)
    v = (mats["v"].astype(np.float64) @ xn.astype(np.float64)).astype(np.float32)
    q = np.concatenate([rms(q[h * HD:(h + 1) * HD], qn.astype(np.float32)) for h in range(NH)])
    k = np.concatenate([rms(k[h * HD:(h + 1) * HD], kn.astype(np.float32)) for h in range(KVH)])
    # position 0: RoPE identity, single cache row -> og = v (head order == natural
    # order: heads are contiguous chunks of QW / KVW-wide blocks)
    og = np.concatenate([v[h * HD:(h + 1) * HD] for h in range(KVH) for _ in range(NH // KVH)])
    out = (mats["o"].astype(np.float64) @ og.astype(np.float64)).astype(np.float32)
    res = xres + out
    xm = rms(res, postln.astype(np.float32))
    Wm = rw.astype(np.float64).reshape(HID, E)  # pool order [HID, E]
    logits = (Wm.T @ xm.astype(np.float64)).astype(np.float32)
    p = (1 / (1 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
    sel = p.astype(np.float64) + bias.astype(np.float64)
    idx = np.argsort(-sel)[:TOPK].astype(np.int32)
    w = p[idx].astype(np.float64)
    w /= w.sum()
    rec = np.zeros(1024, np.float32)
    rec[:E] = p
    rec[E:E + TOPK] = idx.view(np.float32)
    rec[E + 8:E + 8 + TOPK] = w.astype(np.float32)
    print("ref p[:6]", p[:6], "\nref idx", idx, "\nref w", w)

    # ---- consts
    consts = np.zeros(CA_BYTES, np.uint8)

    def put(off, arr, dtype):
        consts[off:off + arr.nbytes] = np.frombuffer(arr.astype(dtype).tobytes(), np.uint8)

    put(CA_LNW, lnw, bfloat16); put(CA_POSTLN, postln, bfloat16)
    put(CA_META, qn, bfloat16); put(CA_META + HD * 2, kn, bfloat16)
    put(CA_RW, rw, bfloat16); put(CA_BIAS, bias, np.float32)
    seq = ptab()
    (HERE / "pool_axsyn.bin").write_bytes(pool.tobytes())
    (HERE / "consts_axsyn.bin").write_bytes(consts.tobytes())
    (HERE / "xres_axsyn.bin").write_bytes(xres.tobytes())
    (HERE / "ptab_axsyn.bin").write_bytes(seq)
    (HERE / "ref_axsyn.bin").write_bytes(rec.tobytes())
    bd = os.environ.get("AX_BUILD_DIR", "build_axsyn")
    cfg = [
        "device",
        f"xclbin X {bd}/final.xclbin",
        f"kernelx ax0 X {bd}/insts.bin",
        f"buf pool {POOL_BYTES} pool_axsyn.bin",
        f"buf xres {HID*4} xres_axsyn.bin",
        f"buf consts {CA_BYTES} consts_axsyn.bin",
        f"buf kv {KV_BYTES}",
        f"buf act {AA_BYTES}",
        f"buf ptab {PTAB_BYTES} ptab_axsyn.bin",
        "attnpos ax0 0",
        f"run ax0 pool xres consts kv act ptab",
        f"dump act rec_axsyn.bin {1024*4} {AA_ROUT}",
        "",
    ]
    (HERE / "run_ax_syn.cfg").write_text("\n".join(cfg), newline="\n")
    print("cfg written; run: run_kernel run_ax_syn.cfg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
