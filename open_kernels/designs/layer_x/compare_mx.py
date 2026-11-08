import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
g = np.fromfile(HERE / "y_mx.bin", np.float32).astype(np.float64)
r = np.fromfile(HERE / "ref_mx.bin", np.float32).astype(np.float64)
rel = np.abs(g - r).max() / (np.abs(r).max() + 1e-30)
cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
ok = rel < 5e-3 and cos > 0.9999 and bool(np.isfinite(g).all())
print(f"{'PASS' if ok else 'FAIL'} mx out cos={cos:.7f} maxrel={rel:.3e}")
print("got", g[:5], "\nref", r[:5])
for c in range(8):
    s = slice(c * 256, (c + 1) * 256)
    rc = np.abs(g[s] - r[s]).max() / (np.abs(r).max() + 1e-30)
    print(f"  core {c}: maxrel {rc:.3e}")
sys.exit(0 if ok else 1)
