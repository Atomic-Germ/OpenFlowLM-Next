"""Compare ln outputs: y (fp32, exact add) and xn (bf16: allow 1-ulp rounding differences)."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
gy = np.fromfile(HERE / "y_y.bin", np.float32).astype(np.float64)
ry = np.fromfile(HERE / "ref_y.bin", np.float32).astype(np.float64)
rel_y = np.abs(gy - ry).max() / (np.abs(ry).max() + 1e-30)
ok_y = rel_y < 1e-6
print(f"{'PASS' if ok_y else 'FAIL'} y  maxrel={rel_y:.3e}")

gx = np.fromfile(HERE / "y_xn.bin", np.uint8).view(bfloat16)
rx = np.fromfile(HERE / "ref_xn.bin", np.uint8).view(bfloat16)
gf, rf = gx.astype(np.float64), rx.astype(np.float64)
rel = np.abs(gf - rf).max() / (np.abs(rf).max() + 1e-30)
cos = float(gf @ rf / (np.linalg.norm(gf) * np.linalg.norm(rf) + 1e-30))
ndiff = int((gx.view(np.uint16) != rx.view(np.uint16)).sum())
ok_x = rel < 8e-3 and cos > 0.999999 and ndiff < len(rx) // 20
print(f"{'PASS' if ok_x else 'FAIL'} xn cos={cos:.8f} maxrel={rel:.3e} bf16-mismatches={ndiff}/{len(rx)}")
sys.exit(0 if (ok_y and ok_x) else 1)
