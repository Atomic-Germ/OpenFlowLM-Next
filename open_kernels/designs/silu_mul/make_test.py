"""Test vectors for silu_mul: random g/u (fp32[512]); fp64 reference h = bf16(silu(g)*u).
No captured buffers needed. Paths in run.cfg are relative to this directory."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
N = 512


def main() -> int:
    rng = np.random.default_rng(0)
    g = (rng.standard_normal(N) * 2).astype(np.float32)
    u = (rng.standard_normal(N) * 2).astype(np.float32)
    g64, u64 = g.astype(np.float64), u.astype(np.float64)
    h = (g64 / (1 + np.exp(-g64)) * u64).astype(np.float32).astype(bfloat16)
    (HERE / "g.bin").write_bytes(g.tobytes())
    (HERE / "u.bin").write_bytes(u.tobytes())
    (HERE / "ref_h.bin").write_bytes(h.tobytes())
    cfg = "\n".join([
        "device",
        "xclbin G build/final.xclbin",
        "kernelx k G build/insts.bin",
        f"buf g {g.nbytes} g.bin",
        f"buf u {u.nbytes} u.bin",
        f"buf h {h.nbytes}",
        "run k g u h",
        "run k g u h",
        f"dump h y_h.bin {h.nbytes}",
        "",
    ])
    (HERE / "run.cfg").write_text(cfg, newline="\n")
    print(f"h[:4]={h[:4].astype(np.float32)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
