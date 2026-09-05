"""Compare moe_combine out (fp32) against the fp64 reference."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
g = np.fromfile(HERE / "y_out.bin", np.float32).astype(np.float64)
r = np.fromfile(HERE / "ref_out.bin", np.float32).astype(np.float64)
rel = np.abs(g - r).max() / (np.abs(r).max() + 1e-30)
cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
ok = rel < 1e-4 and cos > 0.9999999 and np.isfinite(g).all()
print(f"{'PASS' if ok else 'FAIL'} out cos={cos:.9f} maxrel={rel:.3e}")
print("got", g[:5], "\nref", r[:5])
sys.exit(0 if ok else 1)
