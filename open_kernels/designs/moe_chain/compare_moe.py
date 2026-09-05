"""Compare the MoE chain: router selection, then the block output (fp32) vs the
bf16-faithful fp64 reference (and, for scale, vs the replica's fp32-xm version)."""
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
    print(f"{'PASS' if ok else 'FAIL'} {name:8} cos={cos:.7f} maxrel={rel:.3e}")
    return ok


ok = True
gr = np.fromfile(HERE / "y_rout.bin", np.float32)
rr = np.fromfile(HERE / "ref_rout.bin", np.float32)
gi, ri = gr[256:264].view(np.int32).tolist(), rr[256:264].view(np.int32).tolist()
same = gi == ri
print(f"{'PASS' if same else 'FAIL'} router idx got={gi} ref={ri}")
ok &= same
ok &= metrics("xm", np.fromfile(HERE / "y_xm.bin", np.uint8).view(bfloat16), np.fromfile(HERE / "ref_xm.bin", np.float32), 8e-3)
ok &= metrics("out", np.fromfile(HERE / "y_out.bin", np.float32), np.fromfile(HERE / "ref_out.bin", np.float32), 5e-3)
metrics("out~rep", np.fromfile(HERE / "y_out.bin", np.float32), np.fromfile(HERE / "ref_out_replica.bin", np.float32), 5e-2)
g = np.fromfile(HERE / "y_out.bin", np.float32); r = np.fromfile(HERE / "ref_out.bin", np.float32)
print("out got", g[:5], "\nout ref", r[:5])
sys.exit(0 if ok else 1)
