"""Synthetic vectors for the short-conv element: random B/C/u/taps/state, fp64 ref.

    SC_W=512 SC_TAPS=3 python make_test_sc.py ; run_kernel run_sc.cfg ; python compare_sc.py
Scales make the bf16 rounding visible but not the check: B/C/u ~ N(0,0.5),
taps ~ N(0,0.5), state ~ N(0,0.5). Reference mirrors sc.h in fp64.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
SCW = int(os.environ.get("SC_W", 512))
TAPS = int(os.environ.get("SC_TAPS", 3))

def main() -> int:
    rng = np.random.default_rng(0)
    B = (rng.standard_normal(SCW) * 0.5).astype(np.float32)
    C = (rng.standard_normal(SCW) * 0.5).astype(np.float32)
    u = (rng.standard_normal(SCW) * 0.5).astype(np.float32)
    s0 = (rng.standard_normal(SCW) * 0.5).astype(np.float32)
    s1 = (rng.standard_normal(SCW) * 0.5).astype(np.float32)
    w = (rng.standard_normal((3, SCW)) * 0.5).astype(np.float32).astype(bfloat16)
    bcus = np.concatenate([B, C, u, s0, s1]).astype(np.float32)
    (HERE / "bcus.bin").write_bytes(bcus.tobytes())
    (HERE / "w.bin").write_bytes(w.tobytes())
    Bx = B.astype(np.float64) * u.astype(np.float64)
    conv = s0.astype(np.float64) * w[0].astype(np.float64) + s1.astype(np.float64) * w[1].astype(np.float64) + Bx * w[2].astype(np.float64)
    y = (C.astype(np.float64) * conv).astype(np.float32).astype(bfloat16)
    n0 = s1
    n1 = Bx.astype(np.float32)
    (HERE / "ref_y.bin").write_bytes(y.tobytes())
    (HERE / "ref_ns.bin").write_bytes(np.concatenate([n0, n1]).astype(np.float32).tobytes())
    cfg = "\n".join([
        "device",
        "xclbin G build_sc/final.xclbin",
        "kernelx k G build_sc/insts.bin",
        f"buf bcus {bcus.nbytes} bcus.bin",
        f"buf w {w.nbytes} w.bin",
        f"buf y {y.nbytes}",
        f"buf ns {2*SCW*4}",
        "run k bcus w y ns",
        "run k bcus w y ns",
        "dump y y_y.bin %d" % y.nbytes,
        "dump ns y_ns.bin %d" % (2*SCW*4),
        "",
    ])
    (HERE / "run_sc.cfg").write_text(cfg, newline="\n")
    print(f"y[:4]={y[:4].astype(np.float32)} SC_W={SCW} taps={TAPS}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
