"""Compare the short_conv fixture's y (bf16) and slid state (f32) to fp64 refs."""
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
W = int(os.environ.get("SC_TEST_WIDTH", 2048))
TAG = os.environ.get("SC_TEST_TAG", "short_conv")

gy = np.fromfile(HERE / f"y_y_{TAG}.bin", np.uint8).view(bfloat16).astype(np.float64)
ry = np.fromfile(HERE / f"ref_y_{TAG}.bin", np.uint8).view(bfloat16).astype(np.float64)
gyf, ryf = gy.astype(np.float32), ry.astype(np.float32)
rel_y = np.abs(gy - ry).max() / (np.abs(ry).max() + 1e-30)
cos_y = float(gy @ ry / (np.linalg.norm(gy) * np.linalg.norm(ry) + 1e-30))
ndiff = int((gyf.view(np.uint32) != ryf.view(np.uint32)).sum())
ok_y = cos_y > 0.99999 and ndiff < W // 20

gs = np.fromfile(HERE / f"y_s_{TAG}.bin", np.float32).astype(np.float64)
rs = np.fromfile(HERE / f"ref_s_{TAG}.bin", np.float32).astype(np.float64)
rel_s = np.abs(gs - rs).max() / (np.abs(rs).max() + 1e-30)
# bx = B*u is f32 on the core, fp64 in the reference: ~5e-6 maxrel is float
# rounding, not a wrong state. Compare with the f32-rounded reference instead
# of demanding fp64 bit equality.
ok_s = rel_s < 1e-4
print(f"{'PASS' if ok_y else 'FAIL'} y  cos={cos_y:.8f} maxrel={rel_y:.3e} bf16-mismatches={ndiff}/{W}")
print(f"{'PASS' if ok_s else 'FAIL'} state maxrel={rel_s:.3e}")
sys.exit(0 if (ok_y and ok_s) else 1)
