"""Compare router output: probabilities (fp32), top-8 indices (exact set + order) and weights."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
g = np.fromfile(HERE / "y_out.bin", np.float32)
r = np.fromfile(HERE / "ref_out.bin", np.float32)
E = 256
gp, rp = g[:E].astype(np.float64), r[:E].astype(np.float64)
rel = np.abs(gp - rp).max() / (np.abs(rp).max() + 1e-30)
gi, ri = g[E:E + 8].view(np.int32), r[E:E + 8].view(np.int32)
gw, rw = g[E + 8:E + 16].astype(np.float64), r[E + 8:E + 16].astype(np.float64)
same_set = set(gi.tolist()) == set(ri.tolist())
same_order = gi.tolist() == ri.tolist()
wrel = np.abs(gw - rw).max() / (np.abs(rw).max() + 1e-30)
ok = rel < 1e-4 and same_set and wrel < 1e-4 and np.isfinite(g).all()
print(f"{'PASS' if ok else 'FAIL'} p maxrel={rel:.3e} idx got={gi.tolist()} ref={ri.tolist()} "
      f"same_set={same_set} same_order={same_order} w maxrel={wrel:.3e}")
print("w got", np.round(gw, 5).tolist(), "\nw ref", np.round(rw, 5).tolist())
sys.exit(0 if ok else 1)
