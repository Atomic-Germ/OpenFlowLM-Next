"""Compare the whole-layer chain (lx0 -> dn -> lx1 -> lx2) against the fp64 replica
(layer_chain's references and tolerances) and moe_chain's MoE references: the
routing indices and the block output = the layer's output residual."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from layout import A_QKV, A_RES, A_ROUT, A_XM, A_XN, A_Z, S_HEAD_BYTES, STATE_S_OFF  # noqa: E402


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
ok = True
ok &= metrics("xn", act[A_XN:A_XN + 4096].view(bfloat16), np.fromfile(HERE / "ref_xn.bin", np.float32), 8e-3)
lc = HERE.parent / "layer_chain"
if (lc / "y_qkv.bin").is_file():
    ok &= metrics("qkv", act[A_QKV:A_Z].view(np.float32), np.fromfile(lc / "y_qkv.bin", np.float32), 1e-4)
ok &= metrics("xres", act[A_RES:A_RES + 8192].view(np.float32), np.fromfile(HERE / "ref_xres.bin", np.float32), 2e-2)
ok &= metrics("xm", act[A_XM:A_XM + 4096].view(bfloat16), np.fromfile(HERE / "ref_xm.bin", np.float32), 2e-2)
ystate = np.fromfile(HERE / "y_state.bin", np.uint8)
sp = ystate[STATE_S_OFF:].reshape(32, S_HEAD_BYTES)
ok &= metrics("S", sp[:, :128 * 512].reshape(-1).view(np.float32), np.fromfile(HERE / "ref_S.bin", np.float32), 2e-2)
pad = sp[:, 128 * 512:].view(np.float32)
print(f"{'PASS' if not pad.any() else 'FAIL'} S pad rows stay zero (absmax {np.abs(pad).max():.3g})")
ok &= not pad.any()
ok &= metrics("nstate", ystate[:49152].view(bfloat16), np.fromfile(HERE / "ref_cs.bin", np.float32), 2e-2)
idx = act[A_ROUT + 1024:A_ROUT + 1056].view(np.int32)
ref_idx = np.fromfile(HERE / "ref_rout.bin", np.uint8)[1024:1056].view(np.int32)
same = idx.tolist() == ref_idx.tolist()
print(f"{'PASS' if same else 'FAIL'} routing {idx.tolist()} ref {ref_idx.tolist()}")
ok &= same
ok &= metrics("out", np.fromfile(HERE / "y_xres.bin", np.float32), np.fromfile(HERE / "ref_out.bin", np.float32), 5e-3)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
