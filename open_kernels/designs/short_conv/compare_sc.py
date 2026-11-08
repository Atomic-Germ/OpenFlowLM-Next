"""Compare short-conv element: y (bf16) and state (f32)."""
import os, sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
HERE = Path(__file__).parent
SCW = int(os.environ.get("SC_W", 512))
gy = np.fromfile(HERE / "y_y.bin", np.uint8).view(bfloat16).astype(np.float64)
ry = np.fromfile(HERE / "ref_y.bin", np.uint8).view(bfloat16).astype(np.float64)
rel = np.abs(gy - ry).max() / (np.abs(ry).max() + 1e-30)
cos = float(gy @ ry / (np.linalg.norm(gy) * np.linalg.norm(ry) + 1e-30))
ok_y = rel < 8e-3 and cos > 0.999999
print(f"{'PASS' if ok_y else 'FAIL'} y cos={cos:.8f} maxrel={rel:.3e}")
gs = np.fromfile(HERE / "y_ns.bin", np.float32).astype(np.float64)
rs = np.fromfile(HERE / "ref_ns.bin", np.float32).astype(np.float64)
rel2 = np.abs(gs - rs).max() / (np.abs(rs).max() + 1e-30)
ok_s = rel2 < 1e-5
print(f"{'PASS' if ok_s else 'FAIL'} state maxrel={rel2:.3e}")
sys.exit(0 if (ok_y and ok_s) else 1)
