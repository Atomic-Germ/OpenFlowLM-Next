"""Compare router output generically (E/topk/sigmoid/bias from env)."""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
E = int(os.environ.get("ROUTER_E", 256))
TOPK = int(os.environ.get("ROUTER_TOPK", 8))
g = np.fromfile(HERE / "y_out.bin", np.float32)
r = np.fromfile(HERE / "ref_out.bin", np.float32)
gp, rp = g[:E].astype(np.float64), r[:E].astype(np.float64)
rel = np.abs(gp - rp).max() / (np.abs(rp).max() + 1e-30)
gi, ri = g[E:E + 8].view(np.int32), r[E:E + 8].view(np.int32)
gw, rw = g[E + 8:E + 8 + TOPK].astype(np.float64), r[E + 8:E + 8 + TOPK].astype(np.float64)
same_set = set(gi[:TOPK].tolist()) == set(ri[:TOPK].tolist())
same_order = gi[:TOPK].tolist() == ri[:TOPK].tolist()
wrel = np.abs(gw - rw).max() / (np.abs(rw).max() + 1e-30)
ok = rel < 1e-3 and same_set and wrel < 1e-3 and np.isfinite(g).all()
print(f"{'PASS' if ok else 'FAIL'} p maxrel={rel:.3e} idx got={gi[:TOPK].tolist()} ref={ri[:TOPK].tolist()} "
      f"same_set={same_set} same_order={same_order} w maxrel={wrel:.3e}")
print("w got", np.round(gw, 5).tolist(), "\nw ref", np.round(rw, 5).tolist())
sys.exit(0 if ok else 1)
