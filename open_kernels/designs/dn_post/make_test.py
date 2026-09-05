"""Test vectors for dn_post: random o/z, real ssm_norm weight from the captured L0
side pool ($OPEN_KERNELS_CAPS/m0d/000119.bo @65536, bf16[128]); fp64 reference.
Paths in run.cfg are relative to this directory."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402

D, HD = 4096, 128


def main() -> int:
    raw = np.fromfile(FX.caps("m0d/000119.bo"), np.uint8)
    nw = raw[65536:65536 + 256].view(bfloat16).copy()
    nwp = np.zeros(2048, bfloat16)
    nwp[:HD] = nw
    rng = np.random.default_rng(0)
    o = (rng.standard_normal(D) * 0.3).astype(np.float32)
    z = (rng.standard_normal(D) * 1.5).astype(np.float32)
    o64 = o.astype(np.float64).reshape(32, HD)
    on = o64 / np.sqrt((o64 ** 2).mean(-1, keepdims=True) + 1e-6)
    z64 = z.astype(np.float64)
    og = ((on * nw.astype(np.float64)).reshape(D) * (z64 / (1 + np.exp(-z64)))).astype(np.float32).astype(bfloat16)
    (HERE / "o.bin").write_bytes(o.tobytes())
    (HERE / "z.bin").write_bytes(z.tobytes())
    (HERE / "nw.bin").write_bytes(nwp.tobytes())
    (HERE / "ref_og.bin").write_bytes(og.tobytes())
    cfg = "\n".join([
        "device",
        "xclbin G build/final.xclbin",
        "kernelx k G build/insts.bin",
        f"buf o {o.nbytes} o.bin",
        f"buf z {z.nbytes} z.bin",
        f"buf nw {nwp.nbytes} nw.bin",
        f"buf og {og.nbytes}",
        "run k o z nw og",
        "run k o z nw og",
        f"dump og y_og.bin {og.nbytes}",
        "",
    ])
    (HERE / "run.cfg").write_text(cfg, newline="\n")
    print(f"og[:4]={og[:4].astype(np.float32)} absmax={np.abs(og.astype(np.float32)).max():.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
