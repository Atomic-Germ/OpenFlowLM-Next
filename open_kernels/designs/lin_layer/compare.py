"""Compare the fused lin_a -> dn -> lin_c chain against the fp64 CPU replica
(same references and tolerances as layer_chain/compare_chain.py; xn/og are
rounded to bf16 as FLM does, so residual-level agreement is ~1e-3)."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from layout import A_QKV, A_XN, A_Z, H_XM, H_XRES  # noqa: E402


def metrics(name, g, r, tol):
    g = g.astype(np.float64).ravel()
    r = r.astype(np.float64).ravel()
    rel = np.abs(g - r).max() / (np.abs(r).max() + 1e-30)
    cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    fin = bool(np.isfinite(g).all())
    ok = fin and rel < tol and cos > 0.9999
    print(f"{'PASS' if ok else 'FAIL'} {name:7} cos={cos:.7f} maxrel={rel:.3e} finite={fin}")
    return ok


act = np.fromfile(HERE / "y_act.bin", np.uint8)
hdr = np.fromfile(HERE / "y_hdr.bin", np.uint8)
ok = True
ok &= metrics("xn", act[A_XN:A_XN + 4096].view(bfloat16), np.fromfile(HERE / "ref_xn.bin", np.float32), 8e-3)
lc = HERE.parent / "layer_chain"
if (lc / "y_qkv.bin").is_file():       # the unfused chain's own qkv (same kernel): should be ~exact
    ok &= metrics("qkv", act[A_QKV:A_Z].view(np.float32), np.fromfile(lc / "y_qkv.bin", np.float32), 1e-4)
ok &= metrics("xres", hdr[H_XRES:H_XRES + 8192].view(np.float32), np.fromfile(HERE / "ref_xres.bin", np.float32), 2e-2)
ok &= metrics("xm", hdr[H_XM:H_XM + 4096].view(bfloat16), np.fromfile(HERE / "ref_xm.bin", np.float32), 2e-2)
ok &= metrics("S", np.fromfile(HERE / "y_S.bin", np.float32), np.fromfile(HERE / "ref_S.bin", np.float32), 2e-2)
ok &= metrics("nstate", np.fromfile(HERE / "y_state.bin", np.uint8).view(bfloat16), np.fromfile(HERE / "ref_cs.bin", np.float32), 2e-2)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
