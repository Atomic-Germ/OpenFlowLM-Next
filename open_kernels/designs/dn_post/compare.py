"""Compare dn_post og (bf16): allow 1-ulp bf16 rounding differences vs the fp64 reference."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
g = np.fromfile(HERE / "y_og.bin", np.uint8).view(bfloat16)
r = np.fromfile(HERE / "ref_og.bin", np.uint8).view(bfloat16)
gf, rf = g.astype(np.float64), r.astype(np.float64)
rel = np.abs(gf - rf).max() / (np.abs(rf).max() + 1e-30)
cos = float(gf @ rf / (np.linalg.norm(gf) * np.linalg.norm(rf) + 1e-30))
ndiff = int((g.view(np.uint16) != r.view(np.uint16)).sum())
ok = rel < 8e-3 and cos > 0.999999 and ndiff < len(r) // 20
print(f"{'PASS' if ok else 'FAIL'} og cos={cos:.8f} maxrel={rel:.3e} bf16-mismatches={ndiff}/{len(r)} finite={np.isfinite(gf).all()}")
print("got", gf[:6], "\nref", rf[:6])
sys.exit(0 if ok else 1)
