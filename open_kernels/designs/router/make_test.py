"""Test vectors for router: xm = the layer-chain's MoE input if present (ref_xm.bin,
rounded to bf16) else random; W = real moe_router from the captured L0 pack
($OPEN_KERNELS_CAPS/m0d/000118.bo @12288, bf16 [2048][256]); fp64 reference.
Paths in run.cfg are relative to this directory."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402

CHAIN_XM = HERE.parent / "layer_chain" / "ref_xm.bin"
HID, E = 2048, 256


def main() -> int:
    W = np.fromfile(FX.caps("m0d/000118.bo"), np.uint8)[12288:12288 + HID * E * 2].view(bfloat16).reshape(HID, E)
    if CHAIN_XM.is_file():
        xm = np.fromfile(CHAIN_XM, np.float32).astype(bfloat16)
    else:
        xm = (np.random.default_rng(0).standard_normal(HID) * 0.5).astype(np.float32).astype(bfloat16)
    lg = xm.astype(np.float64) @ W.astype(np.float64)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    top = np.argsort(-p, kind="stable")[:8]
    w8 = p[top] / p[top].sum()
    ref = np.zeros(1024, np.float32)
    ref[:E] = p
    ref[E:E + 8] = top.astype(np.int32).view(np.float32)
    ref[E + 8:E + 16] = w8
    (HERE / "xm.bin").write_bytes(xm.tobytes())
    (HERE / "w.bin").write_bytes(W.tobytes())
    (HERE / "ref_out.bin").write_bytes(ref.tobytes())
    cfg = "\n".join([
        "device",
        "xclbin G build/final.xclbin",
        "kernelx k G build/insts.bin",
        f"buf xm {xm.nbytes} xm.bin",
        f"buf w {W.nbytes} w.bin",
        "buf out 4096",
        "run k xm w out",
        "run k xm w out",
        "dump out y_out.bin 4096",
        "",
    ])
    (HERE / "run.cfg").write_text(cfg, newline="\n")
    print(f"top8={top.tolist()} w8={np.round(w8, 4).tolist()} p.max={p.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
