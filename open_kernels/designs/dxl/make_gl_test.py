r"""Test vectors for gl (the L-row GEMV): random pool-order q4_1 weights, L random
activations, the fp64 reference per token from the same bytes.

    python make_gl_test.py --n 6144 --k 4096 --l 4 [--f32] [--runs 3] [--build build_gl]

Writes w.bin, x.bin, ref.bin and run.cfg here (paths relative to this directory), and
x_tok<j>.bin (bf16) per token so the one-token gemv_q4 design can be run on each for the
bit-identity check (compare_gl.py --against-gemv).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from q4_1_pack import pack_q4_1_pool, pool_reference, random_q4_1_blocks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6144)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--l", type=int, default=4)
    ap.add_argument("--f32", action="store_true", help="fp32 activations (the down projection's h)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", default="build_gl")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    w = pack_q4_1_pool(random_q4_1_blocks(a.n, a.k, rng), 2)
    # a different scale per token, so a token reading another's table fails loudly
    xs = [(rng.standard_normal(a.k) * (0.5 + j)).astype(np.float32) for j in range(a.l)]
    xb = [x.astype(bfloat16) for x in xs]
    # the kernel rounds an fp32 input to bf16 before quantising, so the reference uses bf16 either way
    ref = np.stack([pool_reference(w, xb[j], a.n, a.k, 2) for j in range(a.l)]).astype(np.float32)
    x = np.stack(xs) if a.f32 else np.stack(xb)
    (HERE / "w.bin").write_bytes(w.tobytes())
    (HERE / "x.bin").write_bytes(x.tobytes())
    (HERE / "ref.bin").write_bytes(ref.tobytes())
    for j in range(a.l):
        (HERE / f"x_tok{j}.bin").write_bytes(xb[j].tobytes())
    cfg = ["device", f"xclbin G {a.build}/final.xclbin", f"kernelx k G {a.build}/insts.bin",
           f"buf w {w.nbytes} w.bin", f"buf x {x.nbytes} x.bin", f"buf y {ref.nbytes}"]
    cfg += ["run k w x y"] * a.runs
    cfg += [f"dump y y.bin {ref.nbytes}", ""]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"N={a.n} K={a.k} L={a.l} f32={a.f32}: w {w.nbytes} B, x {x.nbytes} B, ref absmax {np.abs(ref).max():.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
