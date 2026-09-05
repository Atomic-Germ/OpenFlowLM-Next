"""Compare y_s.bin / y_o.bin against ref_s.bin / ref_o.bin (fp64 metrics)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def check(name: str, got_f: str, ref_f: str, tol_rel: float) -> bool:
    got = np.fromfile(HERE / got_f, np.float32).astype(np.float64)
    ref = np.fromfile(HERE / ref_f, np.float32).astype(np.float64)
    n = min(len(got), len(ref))
    got, ref = got[:n], ref[:n]
    scale = np.abs(ref).max() + 1e-30
    rel = np.abs(got - ref).max() / scale
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    bad = np.flatnonzero(np.abs(got - ref) > 1e-3 * scale)
    ok = cos > 0.9999999 and rel < tol_rel and bool(np.isfinite(got).all())
    print(f"{'PASS' if ok else 'FAIL'} {name} n={n} cos={cos:.9f} maxrel={rel:.3e} "
          f"finite={np.isfinite(got).all()} nbad={len(bad)}")
    if len(bad):
        print("  first bad idx:", bad[:12])
        print("  got:", got[bad[:6]])
        print("  ref:", ref[bad[:6]])
    return ok


ok_s = check("S_out", "y_s.bin", "ref_s.bin", 1e-4)
ok_o = check("o", "y_o.bin", "ref_o.bin", 1e-4)
sys.exit(0 if (ok_s and ok_o) else 1)
