"""Compare dn_glue outputs: new conv state (bit-exact expected) and per-head records
(fp32 throughout: k/q/v ~1e-5, decay/beta ~1e-7)."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
ok = True

got = np.fromfile(HERE / "y_nstate.bin", np.uint8).view(bfloat16)
ref = np.fromfile(HERE / "ref_nstate.bin", np.uint8).view(bfloat16)
neq = int((got.view(np.uint16) != ref.view(np.uint16)).sum())
print(f"{'PASS' if neq == 0 else 'FAIL'} nstate bit-exact: {neq} of {len(ref)} differ")
ok &= neq == 0

gv = np.fromfile(HERE / "y_vec.bin", np.float32).reshape(32, 512).astype(np.float64)
rv = np.fromfile(HERE / "ref_vec.bin", np.float32).reshape(32, 512).astype(np.float64)


def field(name, g, r, tol):
    global ok
    scale = np.abs(r).max() + 1e-30
    rel = np.abs(g - r).max() / scale
    cos = float((g.ravel() @ r.ravel()) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    fin = bool(np.isfinite(g).all())
    good = fin and rel < tol and cos > 0.99999
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name:6} cos={cos:.8f} maxrel={rel:.3e} finite={fin}")


field("k", gv[:, :128], rv[:, :128], 1e-4)
field("q", gv[:, 128:256], rv[:, 128:256], 1e-4)
field("v", gv[:, 256:384], rv[:, 256:384], 1e-4)
field("decay", gv[:, 384], rv[:, 384], 1e-4)
field("beta", gv[:, 385], rv[:, 385], 1e-4)
print("decay got", gv[:4, 384], "ref", rv[:4, 384])
sys.exit(0 if ok else 1)
