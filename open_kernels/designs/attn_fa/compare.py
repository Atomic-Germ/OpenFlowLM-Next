r"""Score attn_fa's o.bin against make_test.py's fp64 reference.

    python compare.py <dir> [--gate 4e-3]

rel_fro over the whole block, the worst row's cosine, and the worst head / token, so a layout error
(one head or one core's rows wrong) shows as a localized failure rather than a uniform one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--gate", type=float, default=4e-3)
    ap.add_argument("--heads", type=int, default=40)
    a = ap.parse_args()
    ref = np.fromfile(a.dir / "ref.bin", np.float32).astype(np.float64)
    out = np.fromfile(a.dir / "o.bin", np.float32).astype(np.float64)
    if out.size != ref.size:
        print(f"FAIL: o.bin has {out.size} values, ref {ref.size}")
        return 1
    T = 256
    r, o = ref.reshape(T, a.heads, -1), out.reshape(T, a.heads, -1)
    nan = int(np.isnan(o).sum())
    rel = float(np.linalg.norm(o - r) / np.linalg.norm(r))
    cos = (o * r).sum(2) / (np.linalg.norm(o, axis=2) * np.linalg.norm(r, axis=2) + 1e-30)
    t, h = np.unravel_index(np.argmin(cos), cos.shape)
    per_head = np.linalg.norm(o - r, axis=(0, 2)) / np.linalg.norm(r, axis=(0, 2))
    per_core = [float(np.linalg.norm(o[i * 32:(i + 1) * 32] - r[i * 32:(i + 1) * 32]) /
                      np.linalg.norm(r[i * 32:(i + 1) * 32])) for i in range(8)]
    print(f"rel_fro {rel:.3e}, min row cos {cos.min():.6f} (token {t}, head {h}), max |diff| "
          f"{np.abs(o - r).max():.3e}, NaN {nan}")
    print("per head rel_fro: worst " + ", ".join(f"h{i} {per_head[i]:.2e}" for i in np.argsort(-per_head)[:4]))
    print("per core (32 tokens) rel_fro: " + ", ".join(f"{x:.2e}" for x in per_core))
    ok = rel < a.gate and nan == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
