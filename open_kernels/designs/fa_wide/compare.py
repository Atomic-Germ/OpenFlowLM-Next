r"""Check an fa_wide run (make_test.py's o.bin) against float64 attention with the same offset, window and scale.

    python compare.py <testdir> [--rows-sample 256]

PASS when every sampled head's rel_fro is within 3e-2, the gate dit_fa's arithmetic is held to (bf16 Q, K, P, O and the
hardware exp2; designs/dit_fa/README.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

GATE_REL_FRO = 3e-2


def bf(x):
    return x.view(bfloat16).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("testdir")
    ap.add_argument("--rows-sample", type=int, default=256)
    a = ap.parse_args()
    d = Path(a.testdir)
    r = np.load(d / "ref.npz")
    D, L, H, KVH, p0, W, sc = (int(r["D"]), int(r["L"]), int(r["heads"]), int(r["kv_heads"]), int(r["pos0"]),
                               int(r["window"]), float(r["scale"]))
    q = bf(r["q"]).reshape(L, H * D)
    kv = bf(r["kv"]).reshape(-1, 2 * KVH * D)
    k, v = kv[:, :KVH * D], kv[:, KVH * D:]
    o = np.fromfile(d / "o.bin", dtype=bfloat16).astype(np.float64).reshape(L, H * D)
    rows = np.unique(np.linspace(0, L - 1, min(a.rows_sample, L)).astype(int))
    pos = p0 + rows
    cols = np.arange(p0 + L)[None, :]
    mask = cols > pos[:, None]
    if W:
        mask |= cols <= pos[:, None] - W
    worst, ok = 0.0, bool(np.isfinite(o).all())
    for h in range(H):
        kh = h // (H // KVH)
        s = q[rows, h * D:(h + 1) * D] @ k[:p0 + L, kh * D:(kh + 1) * D].T * sc
        s = np.where(mask, -np.inf, s)
        pr = np.exp(s - s.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        ref = pr @ v[:p0 + L, kh * D:(kh + 1) * D]
        got = o[rows, h * D:(h + 1) * D]
        rf = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        worst = max(worst, rf)
        print(f"  head {h:2d}: rel_fro {rf:.3e}")
    verdict = ok and worst <= GATE_REL_FRO
    print(f"{'PASS' if verdict else 'FAIL'}: worst rel_fro {worst:.3e} (gate {GATE_REL_FRO:.0e}), finite {ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
