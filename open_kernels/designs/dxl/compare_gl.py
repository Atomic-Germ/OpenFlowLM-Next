"""Score gl's y.bin against ref.bin per token; --against-gemv also requires bit-identity with y_gemv_tok<j>.bin."""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--l", type=int, required=True)
ap.add_argument("--against-gemv", action="store_true")
a = ap.parse_args()

ref = np.fromfile(HERE / "ref.bin", np.float32).reshape(a.l, -1).astype(np.float64)
y = np.fromfile(HERE / "y.bin", np.float32).reshape(a.l, -1)
ok = True
for j in range(a.l):
    g, r = y[j].astype(np.float64), ref[j]
    cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    rel = float(np.abs(g - r).max() / (np.abs(r).max() + 1e-30))
    good = cos > 0.9999999 and rel < 1e-4
    ok &= good
    line = f"token {j}: {'PASS' if good else 'FAIL'} cos={cos:.9f} maxrel={rel:.3e}"
    if a.against_gemv:
        p = HERE / f"y_gemv_tok{j}.bin"
        one = np.fromfile(p, np.float32)
        same = int((one.view(np.uint32) != y[j].view(np.uint32)).sum())
        ok &= same == 0
        line += f"  vs one-token gemv: {same} of {len(one)} values differ"
    print(line)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
