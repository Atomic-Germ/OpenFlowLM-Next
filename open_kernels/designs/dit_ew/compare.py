r"""Check a dit_ew run (make_test.py's Y.out / Z.out) against the float64 reference.

    python compare.py <testdir>

Prints rel_fro per output next to the floor (the reference itself rounded to bf16) and
PASSes at GATE_REL_FRO: a few bf16 roundings of fp32 math, plus the tanh approximation in
SiLU (see ew.cc).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

GATE_REL_FRO = 1e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("testdir")
    a = ap.parse_args()
    d = Path(a.testdir)
    r = np.load(d / "ref.npz")
    ok = True
    for k in ("Y", "Z"):
        if f"ref_{k}" not in r:
            continue
        ref = r[f"ref_{k}"]
        got = np.fromfile(d / f"{k}.out", dtype=bfloat16).astype(np.float64)[:ref.size]
        got = got.reshape(ref.shape)
        floor = np.linalg.norm(ref.astype(np.float32).astype(bfloat16).astype(np.float64) - ref)
        n = np.linalg.norm(ref)
        rf = np.linalg.norm(got - ref) / n
        worst = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
        fin = bool(np.isfinite(got).all())
        passed = fin and rf <= GATE_REL_FRO
        ok &= passed
        print(f"  {str(r['op'])} {k}: rel_fro {rf:.3e} (bf16 floor {floor / n:.3e}), "
              f"max err / max |ref| {worst:.3e}, finite {fin}  {'PASS' if passed else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
