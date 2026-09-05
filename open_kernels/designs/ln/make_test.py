"""Test vectors for ln: random x/add/w, fp64 reference.

    python make_test.py [--captured] [--runs N]

`--captured` takes the norm weight from FLM's captured L0 pack instead
(phlegm's original fixture; $OPEN_KERNELS_CAPS/m0d/000118.bo or LN_PACK=path,
bf16[2048] at offset 0). Paths in run.cfg are relative to this directory.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402

N = 2048


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captured", action="store_true")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.captured:
        pack = Path(os.environ["LN_PACK"]) if os.environ.get("LN_PACK") else FX.caps("m0d/000118.bo")
        w = np.fromfile(pack, np.uint8)[:N * 2].view(bfloat16).copy()
    else:
        w = (1.0 + rng.standard_normal(N) * 0.1).astype(np.float32).astype(bfloat16)
    x = (rng.standard_normal(N) * 0.5).astype(np.float32)
    add = (rng.standard_normal(N) * 0.5).astype(np.float32)
    y = x.astype(np.float64) + add.astype(np.float64)
    xn = (y / np.sqrt((y ** 2).mean() + 1e-6) * w.astype(np.float64)).astype(np.float32).astype(bfloat16)
    (HERE / "x.bin").write_bytes(x.tobytes())
    (HERE / "add.bin").write_bytes(add.tobytes())
    (HERE / "w.bin").write_bytes(w.tobytes())
    (HERE / "ref_y.bin").write_bytes(y.astype(np.float32).tobytes())
    (HERE / "ref_xn.bin").write_bytes(xn.tobytes())
    cfg = ["device",
           "xclbin G build/final.xclbin",
           "kernelx k G build/insts.bin",
           f"buf x {x.nbytes} x.bin",
           f"buf add {add.nbytes} add.bin",
           f"buf w {w.nbytes} w.bin",
           f"buf y {x.nbytes}",
           f"buf xn {w.nbytes}"]
    cfg += ["run k x add w y xn"] * a.runs
    cfg += [f"dump y y_y.bin {x.nbytes}", f"dump xn y_xn.bin {w.nbytes}", ""]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"xn[:4]={xn[:4].astype(np.float32)} rms={np.sqrt((y**2).mean()):.4f} "
          f"w={'captured' if a.captured else 'random'}")
    print("build: python build_design.py designs/ln/ln.py")
    print("run:   (cd designs/ln && run_kernel run.cfg && python compare.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
