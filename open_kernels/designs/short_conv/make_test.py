"""Test vectors for sc.h at any hidden width, with an fp64 reference.

    SC_TEST_WIDTH=1536 python make_test.py

The reference is model/replica_lfm2.py's short_conv_step, written out here so
the gate needs no model weights: bx = B*u, conv = s0*w0 + s1*w1 + bx*w2
(newest tap LAST), y = bf16(C*conv), and the state slides to [s1, bx].
Paths in run_<tag>.cfg are relative to this directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402

W = int(os.environ.get("SC_TEST_WIDTH", 2048))
TAG = os.environ.get("SC_TEST_TAG", "short_conv")
BUILD = os.environ.get("SC_TEST_BUILD", "build")


def main() -> int:
    if W % 512 or W <= 0:
        print(f"SC_TEST_WIDTH={W}: must be a positive multiple of 512", file=sys.stderr)
        return 1
    rng = np.random.default_rng(0)
    b = (rng.standard_normal(W) * 0.5).astype(np.float32)
    c = (rng.standard_normal(W) * 0.5).astype(np.float32)
    u = (rng.standard_normal(W) * 0.5).astype(np.float32)
    w = [(rng.standard_normal(W) * 0.3).astype(np.float32).astype(bfloat16) for _ in range(3)]
    s0 = (rng.standard_normal(W) * 0.4).astype(np.float32)
    s1 = (rng.standard_normal(W) * 0.4).astype(np.float32)

    bd, cd, ud = b.astype(np.float64), c.astype(np.float64), u.astype(np.float64)
    wd = [x.astype(np.float64) for x in w]
    bx = bd * ud
    conv = s0.astype(np.float64) * wd[0] + s1.astype(np.float64) * wd[1] + bx * wd[2]
    ref_y = (cd * conv).astype(bfloat16)
    ref_s = np.concatenate([s1, bx.astype(np.float32)])

    def put(name, arr):
        (HERE / f"{name}_{TAG}.bin").write_bytes(arr.tobytes())

    # The core has 2 input / 2 output DMA channels: B|C|u share one DDR vector,
    # the two state rows another, the three taps a third (bf16).
    put("x", np.concatenate([b, c, u]))
    put("s", np.concatenate([s0, s1]))
    put("w", np.concatenate(w))
    put("ref_y", ref_y)
    put("ref_s", ref_s)

    cfg = ["device",
           f"xclbin G {BUILD}/final.xclbin",
           f"kernelx k G {BUILD}/insts.bin",
           f"buf x {(3 * W) * 4} x_{TAG}.bin",
           f"buf s {(2 * W) * 4} s_{TAG}.bin",
           f"buf w {(3 * W) * 2} w_{TAG}.bin",
           f"buf y {W * 2}",
           f"buf sn {(2 * W) * 4}",
           "run k x s w y sn",
           f"dump y y_y_{TAG}.bin {W * 2}",
           f"dump sn y_s_{TAG}.bin {(2 * W) * 4}",
           ""]
    (HERE / f"run_{TAG}.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"SC_TEST_WIDTH={W}: {W // 512} element(s); y[:4]={ref_y[:4].astype(np.float32)}")
    print("build: SC_TEST_WIDTH=%d python build_design.py designs/short_conv/sc_test.py %s" % (W, BUILD))
    print(f"run:   (cd designs/short_conv && run_kernel run_{TAG}.cfg && SC_TEST_WIDTH={W} python compare.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
