"""Compare the fused attention layer against attn_chain's bf16-faithful references."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from layout import AA_KVNEW, AA_OG  # noqa: E402


def metrics(name, g, r, tol):
    g = g.astype(np.float64).ravel(); r = r.astype(np.float64).ravel()
    rel = np.abs(g - r).max() / (np.abs(r).max() + 1e-30)
    cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    ok = rel < tol and cos > 0.9999 and bool(np.isfinite(g).all())
    print(f"{'PASS' if ok else 'FAIL'} {name:9} cos={cos:.7f} maxrel={rel:.3e}")
    return ok


f32 = lambda f: np.fromfile(HERE / f, np.float32)
act = np.fromfile(HERE / "y_act.bin", np.uint8)
hdr = np.fromfile(HERE / "y_hdr.bin", np.uint8)
kvnew = act[AA_KVNEW:AA_KVNEW + 2048].view(bfloat16)
ok = True
ok &= metrics("knew", kvnew[:512], f32("ref_knew.bin"), 1e-2)
ok &= metrics("vnew", kvnew[512:], f32("ref_vnew.bin"), 1e-2)
ok &= metrics("og", act[AA_OG:AA_OG + 8192].view(bfloat16), f32("ref_og.bin"), 2e-2)
ok &= metrics("xres", hdr[12288:20480].view(np.float32), f32("ref_xres.bin"), 1e-2)
ok &= metrics("xm", hdr[0:4096].view(bfloat16), f32("ref_xm.bin"), 2e-2)
metrics("xres~rep", hdr[12288:20480].view(np.float32), f32("ref_xres_replica.bin"), 5e-2)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
