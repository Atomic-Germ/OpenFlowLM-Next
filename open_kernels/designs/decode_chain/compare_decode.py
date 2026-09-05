"""Compare the open-kernel decode step with FLM's captured logits
($OPEN_KERNELS_CAPS/m0c/000905.bo) and the CPU replica."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402

ours_all = np.fromfile(HERE / "y_logits.bin", np.float32)
ours = ours_all[1::2][:124160].astype(np.float64)              # FLM's buffer holds the odd vocab rows
cap = np.fromfile(FX.caps("m0c/000905.bo"), np.float32)[:124160].astype(np.float64)
rep = np.fromfile(HERE / "ref_logits_replica.bin", np.float32).astype(np.float64)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def top(a, n=5):
    return [2 * int(i) + 1 for i in np.argsort(-a)[:n]]


print(f"finite={np.isfinite(ours).all()}  ours vs FLM capture corr={corr(ours, cap):.5f}   "
      f"replica vs capture {corr(rep, cap):.5f}   ours vs replica {corr(ours, rep):.5f}")
print(f"top-5 vocab  ours {top(ours)}  capture {top(cap)}  replica {top(rep)}")
for l in range(3):
    g = np.fromfile(HERE / f"y_res{l}.bin", np.float32).astype(np.float64)
    r = np.fromfile(HERE / f"ref_res{l}.bin", np.float32).astype(np.float64)
    print(f"residual after layer {l}: corr vs replica {corr(g, r):.6f}  maxrel {np.abs(g - r).max() / np.abs(r).max():.2e}")
ok = np.isfinite(ours).all() and corr(ours, cap) > 0.98 and top(ours)[0] == top(cap)[0]
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
