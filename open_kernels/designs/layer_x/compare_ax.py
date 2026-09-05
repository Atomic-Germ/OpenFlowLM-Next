"""Compare the whole full-attention layer (ax0 -> moeroute2 -> ax1 at position 11) against
attn_chain's bf16-faithful references: the new cache row (kv row 11), og, the residual after
attention, the MoE input; the routing for sanity; and the third run against the first."""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
AC = HERE.parent / "attn_chain"
sys.path.insert(0, str(HERE))
from layout import AA_OG, AA_RES, AA_ROUT, AA_XM  # noqa: E402


def metrics(name, g, r, tol):
    g = g.astype(np.float64).ravel(); r = r.astype(np.float64).ravel()
    rel = np.abs(g - r).max() / (np.abs(r).max() + 1e-30)
    cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-30))
    ok = rel < tol and cos > 0.9999 and bool(np.isfinite(g).all())
    print(f"{'PASS' if ok else 'FAIL'} {name:9} cos={cos:.7f} maxrel={rel:.3e}")
    return ok


f32 = lambda f: np.fromfile(AC / f, np.float32)


def check(sfx):
    act = np.fromfile(HERE / f"y_ax_act{sfx}.bin", np.uint8)
    kvnew = np.fromfile(HERE / f"y_ax_kvnew{sfx}.bin", np.uint8).view(bfloat16)
    ok = metrics("knew", kvnew[:512], f32("ref_knew.bin"), 1e-2)
    ok &= metrics("vnew", kvnew[512:], f32("ref_vnew.bin"), 1e-2)
    ok &= metrics("og", act[AA_OG:AA_OG + 8192].view(bfloat16), f32("ref_og.bin"), 2e-2)
    ok &= metrics("xres", act[AA_RES:AA_RES + 8192].view(np.float32), f32("ref_xres.bin"), 1e-2)
    ok &= metrics("xm", act[AA_XM:AA_XM + 4096].view(bfloat16), f32("ref_xm.bin"), 2e-2)
    metrics("xres~rep", act[AA_RES:AA_RES + 8192].view(np.float32), f32("ref_xres_replica.bin"), 5e-2)
    idx = act[AA_ROUT + 1024:AA_ROUT + 1056].view(np.int32).tolist()
    sane = len(set(idx)) == 8 and all(0 <= i < 256 for i in idx)
    out = np.fromfile(HERE / f"y_ax_xres{sfx}.bin", np.float32)
    fin = bool(np.isfinite(out).all()) and np.abs(out).max() > 0
    print(f"{'PASS' if sane else 'FAIL'} routing {idx}   {'PASS' if fin else 'FAIL'} layer output finite (absmax {np.abs(out).max():.3g})")
    return ok and sane and fin, out


ok1, out1 = check("")
print("--- third run (xres, kv reloaded; the patched stream replayed)")
ok3, out3 = check("_r3")
same = np.array_equal(out1, out3)
print(f"{'PASS' if same else 'FAIL'} run 3 output == run 1 output")
ok = ok1 and ok3 and same
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
