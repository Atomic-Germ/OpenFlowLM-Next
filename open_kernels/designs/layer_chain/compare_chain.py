"""Compare the open-kernel layer chain against the fp64 CPU replica.

The chain rounds xn and og to bf16 (as FLM does), so residual-level agreement is
~1e-3 relative, not fp32-exact; the recurrent state S carries the same error."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent


def metrics(name, g, r, tol):
    g = g.astype(np.float64).ravel()
    r = r.astype(np.float64).ravel()
    rel = np.abs(g - r).max() / (np.abs(r).max() + 1e-30)
    cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    corr = float(np.corrcoef(g, r)[0, 1]) if len(g) > 1 else cos
    fin = bool(np.isfinite(g).all())
    ok = fin and rel < tol and cos > 0.9999
    print(f"{'PASS' if ok else 'FAIL'} {name:7} cos={cos:.7f} corr={corr:.7f} maxrel={rel:.3e} finite={fin}")
    return ok


ok = True
ok &= metrics("xn", np.fromfile(HERE / "y_xn.bin", np.uint8).view(bfloat16), np.fromfile(HERE / "ref_xn.bin", np.float32), 8e-3)
ok &= metrics("xres", np.fromfile(HERE / "y_xres.bin", np.float32), np.fromfile(HERE / "ref_xres.bin", np.float32), 2e-2)
ok &= metrics("xm", np.fromfile(HERE / "y_xm.bin", np.uint8).view(bfloat16), np.fromfile(HERE / "ref_xm.bin", np.float32), 2e-2)
ok &= metrics("S", np.fromfile(HERE / "y_S.bin", np.float32), np.fromfile(HERE / "ref_S.bin", np.float32), 2e-2)
ok &= metrics("nstate", np.fromfile(HERE / "y_nstate.bin", np.uint8).view(bfloat16), np.fromfile(HERE / "ref_cs.bin", np.float32), 2e-2)
xr = np.fromfile(HERE / "y_xres.bin", np.float32)
rr = np.fromfile(HERE / "ref_xres.bin", np.float32)
print("xres got", xr[:5], "\nxres ref", rr[:5])
sys.exit(0 if ok else 1)
