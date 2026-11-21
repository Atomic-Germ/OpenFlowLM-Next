"""Score the open engine's logits (open_qwen36_cli --dump-logits) against replica_gemma4.py's (2026-10-16).

    python compare_gemma4.py <prefix>_t0.bin <replica.npz> [--key logits_l6]

Both are the soft-capped logits of the prompt's last position. Prints the correlation, the largest
difference, the argmax and the top-5 overlap; exits non-zero below the corr bar.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npu")
    ap.add_argument("ref")
    ap.add_argument("--key", default="logits")
    ap.add_argument("--min-corr", type=float, default=0.999)
    a = ap.parse_args()
    ref = np.load(a.ref)[a.key].astype(np.float64)
    npu = np.fromfile(a.npu, dtype=np.float32).astype(np.float64)[: ref.size]
    corr = float(np.corrcoef(npu, ref)[0, 1])
    t5n, t5r = list(np.argsort(-npu)[:5]), list(np.argsort(-ref)[:5])
    print(f"{a.key}: corr {corr:.6f}, max |npu - ref| {np.abs(npu - ref).max():.4f} "
          f"(ref range {ref.min():.2f}..{ref.max():.2f}), argmax npu {t5n[0]} ref {t5r[0]}, "
          f"top-5 overlap {len(set(t5n) & set(t5r))}/5")
    print("  npu top5", [(int(t), round(float(npu[t]), 3)) for t in t5n])
    print("  ref top5", [(int(t), round(float(ref[t]), 3)) for t in t5r])
    return 0 if corr >= a.min_corr and t5n[0] == t5r[0] else 1


if __name__ == "__main__":
    sys.exit(main())
