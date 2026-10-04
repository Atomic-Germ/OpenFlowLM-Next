r"""Test vectors for probe 0a: one random ternary matrix, packed as 2-bit chunks AND
as q4_1 chunks (q = code, d = s, m = -s), one activation, one fp64 reference.

    python make_test.py --shape gate|down [--runs 20]

Shapes are the 27B's: gate = [17408, 5120] (gate / up), down = [5120, 8192] (the
first half of the split down projection). Writes w_{t2,q4}_<shape>.bin,
x_<shape>.bin, ref_<shape>.bin and run_{t2,q4}_<shape>[_null].cfg here, and prints
the four build commands. `probe.py <shape>` times and checks them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from t2_pack import pack_q4_1_from_ternary, pack_t2_pool, random_pq2_codes, reference  # noqa: E402

SHAPES = {"gate": (17408, 5120), "down": (5120, 8192), "small": (2048, 2048)}
RS = 2


def fwht_blocks(x: np.ndarray, block: int = 1024) -> np.ndarray:
    """Normalized Sylvester Walsh-Hadamard over each `block` of x (PrismML's runtime fwht)."""
    y = x.reshape(-1, block).copy()
    h = 1
    while h < block:
        y = y.reshape(-1, block // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :], y[:, :, 1, :] = a + b, a - b
        h *= 2
    return (y.reshape(-1) / np.sqrt(block))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="gate", choices=sorted(SHAPES))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n, k = SHAPES[a.shape]
    rng = np.random.default_rng(a.seed)
    codes, s = random_pq2_codes(n, k, rng)
    x = rng.standard_normal(k).astype(np.float32).astype(bfloat16)
    ref = reference(codes, s, x)
    # t2h's input: each 1024 block of x through H/32, rounded to bf16 (what the prep feeds the table)
    xr = fwht_blocks(x.astype(np.float64)).astype(np.float32).astype(bfloat16)
    (HERE / f"ref_{a.shape}_h.bin").write_bytes(reference(codes, s, xr).tobytes())
    w = {"t2": pack_t2_pool(codes, s, RS), "q4": pack_q4_1_from_ternary(codes, s, RS)}
    (HERE / f"x_{a.shape}.bin").write_bytes(x.tobytes())
    (HERE / f"ref_{a.shape}.bin").write_bytes(ref.tobytes())
    for fmt, wb in w.items():
        (HERE / f"w_{fmt}_{a.shape}.bin").write_bytes(wb.tobytes())
        for null, kind in [(n_, f_) for f_ in ([fmt, "t2g", "t2h", "t2p", "t2p8", "t2p5", "t2p1"] if fmt == "t2" else [fmt]) for n_ in ("", "_null")]:
            tag = f"{kind}_{a.shape}{null}"
            cfg = ["device", f"xclbin G build_{tag}/final.xclbin", f"kernelx k G build_{tag}/insts.bin",
                   f"buf w {wb.nbytes} w_{fmt}_{a.shape}.bin", f"buf x {x.nbytes} x_{a.shape}.bin",
                   f"buf y {ref.nbytes}"]
            cfg += ["run k w x y"] * a.runs
            cfg += [f"dump y y_{tag}.bin {ref.nbytes}", ""]
            (HERE / f"run_{tag}.cfg").write_text("\n".join(cfg), newline="\n")
            print(f"build: GEMV_FMT={kind} GEMV_N={n} GEMV_K={k} GEMV_NULL={1 if null else 0} "
                  f"python build_design.py designs/gemv_t2/gemv_t2.py designs/gemv_t2/build_{tag}")
    print(f"{a.shape}: N={n} K={k}  t2 {w['t2'].nbytes} B  q4 {w['q4'].nbytes} B  "
          f"({w['q4'].nbytes / w['t2'].nbytes:.3f}x)  ref absmax {np.abs(ref).max():.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
