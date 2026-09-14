"""Compare c_<tag>.bin against ref_<tag>.bin for attn_gemm (float64 metrics).

    python compare.py s2048
    python compare.py pv2048

Gate: rel_fro <= 5e-3, the bf16 x bf16 -> fp32 GEMM's own bar (gemm_q4_prefill/compare.py
says why it is this and not gemv_q4's). The per-row cosine is the attention-relevant
number: a row is one (head, token) pair's scores or its output.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REL_FRO_GATE = 5e-3
M = 2048


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "s2048"
    got = np.fromfile(HERE / f"c_{tag}.bin", np.float32).astype(np.float64)
    ref = np.fromfile(HERE / f"ref_{tag}.bin", np.float32).astype(np.float64)
    n = min(len(got), len(ref))
    got, ref = got[:n], ref[:n]
    rel_fro = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    ok = rel_fro <= REL_FRO_GATE and bool(np.isfinite(got).all())
    print(f"{'PASS' if ok else 'FAIL'} {tag} n={n} rel_fro={rel_fro:.3e} (gate {REL_FRO_GATE:.0e}) "
          f"cos={cos:.9f} finite={np.isfinite(got).all()}")
    if n % M == 0:
        G, R = got.reshape(M, -1), ref.reshape(M, -1)
        c = np.einsum("ij,ij->i", G, R) / (np.linalg.norm(G, axis=1) * np.linalg.norm(R, axis=1) + 1e-30)
        print(f"per-row cosine: min={c.min():.9f} (row {int(np.argmin(c))}) mean={c.mean():.9f}")
        # a transposed C would show up here as a row-cosine collapse with a good overall cosine
        if c.min() < 0.99 and cos > 0.99:
            print("  rows do not match but the whole does: C may be [N, M] -- check the layout")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
