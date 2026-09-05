"""Compare silu_mul h (bf16) vs the fp64->bf16 reference (1-ulp differences allowed)."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
g = np.fromfile(HERE / "y_h.bin", np.uint8).view(bfloat16)
r = np.fromfile(HERE / "ref_h.bin", np.uint8).view(bfloat16)
gf, rf = g.astype(np.float64), r.astype(np.float64)
rel = np.abs(gf - rf).max() / (np.abs(rf).max() + 1e-30)
cos = float(gf @ rf / (np.linalg.norm(gf) * np.linalg.norm(rf) + 1e-30))
nd = int((g.view(np.uint16) != r.view(np.uint16)).sum())
ok = rel < 8e-3 and cos > 0.999999 and nd < len(r) // 20 and np.isfinite(gf).all()
print(f"{'PASS' if ok else 'FAIL'} h cos={cos:.8f} maxrel={rel:.3e} bf16-mismatches={nd}/{len(r)}")
sys.exit(0 if ok else 1)
