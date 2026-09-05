"""Compare the attention chain: new cache rows, gated attention output, residual, MoE input."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent


def metrics(name, g, r, tol):
    g = g.astype(np.float64).ravel(); r = r.astype(np.float64).ravel()
    rel = np.abs(g - r).max() / (np.abs(r).max() + 1e-30)
    cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    ok = rel < tol and cos > 0.9999 and bool(np.isfinite(g).all())
    print(f"{'PASS' if ok else 'FAIL'} {name:9} cos={cos:.7f} maxrel={rel:.3e}")
    return ok


bf = lambda f: np.fromfile(HERE / f, np.uint8).view(bfloat16)
f32 = lambda f: np.fromfile(HERE / f, np.float32)
ok = True
kvnew = bf("y_kvnew.bin")
ok &= metrics("knew", kvnew[:512], f32("ref_knew.bin"), 1e-2)
ok &= metrics("vnew", kvnew[512:], f32("ref_vnew.bin"), 1e-2)
ok &= metrics("og", bf("y_og.bin"), f32("ref_og.bin"), 2e-2)
ok &= metrics("xres", f32("y_xres.bin"), f32("ref_xres.bin"), 1e-2)
ok &= metrics("xm", bf("y_xm.bin"), f32("ref_xm.bin"), 2e-2)
metrics("xres~rep", f32("y_xres.bin"), f32("ref_xres_replica.bin"), 5e-2)
print("xres got", f32("y_xres.bin")[:4], "\nxres ref", f32("ref_xres.bin")[:4])
sys.exit(0 if ok else 1)
