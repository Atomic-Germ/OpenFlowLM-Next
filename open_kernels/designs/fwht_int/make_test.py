r"""Vectors (one input kind per block), fwht_ref.py's expected tables and a run cfg per fwht_int build.

    python make_test.py [--k 8192] [--runs 20]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fwht_ref import BLOCK, bf16, fp_tables, int_tables

HERE = Path(__file__).resolve().parent


def tab_bytes(k: int) -> int:          # fwht_int.py's (gemv_t2_tab_bytes)
    return 2 * k + k // 16


def blocks(rng, n: int, f32: bool) -> np.ndarray:
    out = []
    for b in range(n):
        kind = b % 8
        x = rng.standard_normal(BLOCK)
        if kind == 1:
            x[rng.choice(BLOCK, 4, replace=False)] *= 60
            x[rng.integers(BLOCK)] *= 1500
        elif kind == 2:
            x[:] = 0
        elif kind == 3:
            x *= 1e-5
        elif kind == 4:
            x *= 3e3
        elif kind == 5:
            x[:] = 0
            x[rng.integers(BLOCK)] = -0.75
        elif kind == 6:                      # int16 edge: bf16 lands on +-32640, fp32 rounds to 32768 and must saturate
            x = np.sign(x) * (2 - 2 ** -7)
            if f32:
                x = np.sign(x) * (2 - 2 ** -20)
        elif kind == 7:
            g = rng.standard_normal(BLOCK) * 2
            x = g / (1 + np.exp(-g)) * x
        out.append(x)
    x = np.concatenate(out)
    return x.astype(np.float32) if f32 else bf16(x)


def tab_bytes_of(t) -> bytes:
    xi, sh, hi, lo = t
    return xi.tobytes() + sh.astype(np.int32).tobytes() + hi.tobytes() + lo.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    xs = {"bf16": blocks(rng, a.k // BLOCK, False), "f32": blocks(rng, a.k // BLOCK, True)}
    for name, x in xs.items():
        (HERE / f"x_{name}.bin").write_bytes(x.tobytes())
        (HERE / f"tab_int_{name}.bin").write_bytes(tab_bytes_of(int_tables(x)))
        (HERE / f"tab_fp_{name}.bin").write_bytes(tab_bytes_of(fp_tables(x)))
    tb = tab_bytes(a.k)
    for kind, sfx in [(k, "") for k in range(4)] + [(0, "s"), (1, "s")]:     # "s": an -Os build of the int path
        name = "bf16" if kind in (0, 2) else "f32"
        for reps in (1, 33):
            tag = f"k{kind}{sfx}_r{reps}"
            cfg = ["device", f"xclbin G build_{tag}/final.xclbin", f"kernelx k G build_{tag}/insts.bin",
                   f"buf x {xs[name].nbytes} x_{name}.bin", f"buf t {tb}"]
            cfg += ["run k x t"] * a.runs + [f"dump t t_{tag}.bin {tb}", ""]
            (HERE / f"run_{tag}.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"K={a.k}: table {tb} B; x bf16 {xs['bf16'].nbytes} B, f32 {xs['f32'].nbytes} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
