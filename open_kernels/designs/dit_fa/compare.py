r"""Check a dit_fa run (make_test.py's o.bin) against float64 SDPA.

    python compare.py <testdir> [--heads-sample N] [--rows-sample N]

Scores every sampled head on a sample of query rows: rel_fro over the sampled block,
worst row cosine, finiteness. PASS gate (GATE_REL_FRO) is set from the bf16/bfp16
arithmetic this kernel does; see README.md for where the number comes from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

D = 128
GATE_REL_FRO = 3e-2


def bf(x):
    return x.view(bfloat16).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("testdir")
    ap.add_argument("--heads-sample", type=int, default=6)
    ap.add_argument("--rows-sample", type=int, default=512)
    a = ap.parse_args()
    d = Path(a.testdir)
    r = np.load(d / "ref.npz")
    L, H, KVH = int(r["L"]), int(r["heads"]), int(r["kv_heads"])
    causal, valid = bool(r["causal"]), int(r["valid_len"])
    q, k, v = bf(r["q"]), bf(r["k"]), bf(r["v"])
    o = np.fromfile(d / "o.bin", dtype=bfloat16).astype(np.float64).reshape(L, H * D)

    heads = np.unique(np.linspace(0, H - 1, min(a.heads_sample, H)).astype(int))
    rows = np.unique(np.linspace(0, L - 1, min(a.rows_sample, L)).astype(int))
    worst_rf, worst_cos, ok = 0.0, 1.0, bool(np.isfinite(o).all())
    for h in heads:
        kh = h // (H // KVH)
        qh = q[rows, h * D:(h + 1) * D]
        kk, vv = k[:, kh * D:(kh + 1) * D], v[:, kh * D:(kh + 1) * D]
        s = qh @ kk.T / np.sqrt(D)
        cols = np.arange(L)[None, :]
        mask = cols >= valid
        if causal:
            mask = mask | (cols > rows[:, None])
        s = np.where(mask, -np.inf, s)
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        ref = p @ vv
        got = o[rows, h * D:(h + 1) * D]
        rf = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30)
        worst_rf, worst_cos = max(worst_rf, rf), min(worst_cos, cos.min())
        print(f"  head {h:3d}: rel_fro {rf:.3e}  min row cos {cos.min():.5f}")
    verdict = ok and worst_rf <= GATE_REL_FRO
    print(f"{'PASS' if verdict else 'FAIL'}: worst rel_fro {worst_rf:.3e} (gate {GATE_REL_FRO:.0e}), "
          f"worst row cos {worst_cos:.5f}, finite {ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
