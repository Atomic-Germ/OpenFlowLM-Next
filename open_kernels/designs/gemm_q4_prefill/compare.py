"""Compare y_<tag>.bin against ref_<tag>.bin for gemm_q4_prefill (float64 metrics).

    python compare.py <tag>       # tag = <shape>_t<T>, e.g. qkv_t256

GATE: rel_fro <= 5e-3 (Frobenius relative error), NOT gemv_q4/compare.py's
cos>0.9999999 / maxrel<1e-4. That tighter gate is calibrated for gemv_q4's
EXACT int16 x uint8 integer reduction (only the final scale/store touches
bf16); this design's inner reduction is a genuine bf16 x bf16 -> fp32
mac-accumulate GEMM (the SAME datapath NpuEmbeddings/experiments/m5-pretiled-
gemm/gemm_pretiled.py's own run_one() validates against), and that reference
design's own correctness gate is `tol = 5e-3` on rel_fro for plain bf16
(5e-2 for the bfp16-emulated path) -- never tighter. Measured on this build
(qkv, T=256, seed 0): rel_fro=2.19e-3, comfortably inside that bar, while the
gemv_q4-style maxrel gate (2.28e-3 measured) fails outright -- a genuine
miscalibration this task's first correctness run caught, not a kernel bug
(cosine 0.999998 overall, per-token min 0.999995 corroborate "close, bf16-
precision-limited" rather than "wrong"). See gemm_q4_prefill.py's own module
docstring / the task report for the full account.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REL_FRO_GATE = 5e-3


def metrics(got: np.ndarray, ref: np.ndarray):
    n = min(len(got), len(ref))
    got, ref = got[:n], ref[:n]
    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    rel_fro = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    bad = np.flatnonzero(np.abs(got - ref) > 1e-3 * (np.abs(ref).max() + 1e-30))
    ok = rel_fro <= REL_FRO_GATE
    return ok, n, cos, rel, rel_fro, bad


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "qkv_t256"
    got = np.fromfile(HERE / f"y_{tag}.bin", np.float32).astype(np.float64)
    ref = np.fromfile(HERE / f"ref_{tag}.bin", np.float32).astype(np.float64)

    ok, n, cos, rel, rel_fro, bad = metrics(got, ref)
    print(f"{'PASS' if ok else 'FAIL'} overall n={n} rel_fro={rel_fro:.3e} (gate {REL_FRO_GATE:.0e}) "
          f"cos={cos:.9f} maxrel={rel:.3e} finite={np.isfinite(got).all()} nbad={len(bad)}")
    if len(bad):
        print("first bad idx:", bad[:16])
        print("got:", got[bad[:8]])
        print("ref:", ref[bad[:8]])

    # Per-token (column) worst case: infer T and N_WEIGHT from the tag/shapes
    # table isn't available here standalone, so infer T from the ref length
    # and the shape name's N_WEIGHT via a small lookup (kept in sync with
    # make_test.py's SHAPES table).
    SHAPES = {"qkv": 2560, "down_proj": 2560, "up_gate": 8192}
    shape = tag.rsplit("_t", 1)[0]
    n_weight = SHAPES.get(shape)
    if n_weight and n % n_weight == 0:
        T = n // n_weight
        G = got.reshape(n_weight, T)
        R = ref.reshape(n_weight, T)
        cos_per_tok = np.einsum("it,it->t", G, R) / (
            np.linalg.norm(G, axis=0) * np.linalg.norm(R, axis=0) + 1e-30)
        worst = int(np.argmin(cos_per_tok))
        print(f"per-token cosine: min={cos_per_tok.min():.9f} (token {worst}) "
              f"max={cos_per_tok.max():.9f} mean={cos_per_tok.mean():.9f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
