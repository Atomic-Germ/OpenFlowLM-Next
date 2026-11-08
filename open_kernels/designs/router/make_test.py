"""Test vectors for router: synthetic (default) or captured L0 pack.

Synthetic (the gate shape): xm ~ N(0,0.5) bf16[2048], W ~ N(0,0.1) bf16[2048*E],
bias ~ N(0,0.05) f32[E] (only when ROUTER_BIAS=1). fp64 reference implements
the same law the kernel compiles: softmax, or sigmoid + biased selection with
unbiased renormalisation. Shape from env (ROUTER_E/TOPK/SIGMOID/BIAS, as router.py).

Captured (legacy): W from $OPEN_KERNELS_CAPS/m0d/000118.bo @12288 (needs caps).

    ROUTER_E=64 ROUTER_TOPK=4 ROUTER_SIGMOID=1 ROUTER_BIAS=1 python make_test.py
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

HID = 2048
E = int(os.environ.get("ROUTER_E", 256))
TOPK = int(os.environ.get("ROUTER_TOPK", 8))
SIGMOID = int(os.environ.get("ROUTER_SIGMOID", 0))
BIAS = int(os.environ.get("ROUTER_BIAS", 0))


def main() -> int:
    rng = np.random.default_rng(0)
    if os.environ.get("ROUTER_CAPTURED"):
        W = np.fromfile(FX.caps("m0d/000118.bo"), np.uint8)[12288:12288 + HID * E * 2].view(bfloat16).reshape(HID, E)
        if os.environ.get("ROUTER_XM"):
            xm = np.fromfile(os.environ["ROUTER_XM"], np.float32).astype(bfloat16)
        else:
            xm = (rng.standard_normal(HID) * 0.5).astype(np.float32).astype(bfloat16)
    else:
        W = (rng.standard_normal((HID, E)) * 0.1).astype(np.float32).astype(bfloat16)
        xm = (rng.standard_normal(HID) * 0.5).astype(np.float32).astype(bfloat16)
    lg = xm.astype(np.float64) @ W.astype(np.float64)
    if SIGMOID:
        p = 1.0 / (1.0 + np.exp(-lg))
    else:
        p = np.exp(lg - lg.max())
        p /= p.sum()
    bias = (rng.standard_normal(E) * 0.05).astype(np.float64) if BIAS else None
    sel = p + (bias if BIAS else 0.0)
    top = np.argsort(-sel, kind="stable")[:TOPK]
    w = p[top] / p[top].sum()
    ref = np.zeros(1024, np.float32)
    ref[:E] = p
    ref[E:E + 8] = 0
    ref[E:E + TOPK] = top.astype(np.int32).view(np.float32)
    ref[E + 8:E + 8 + TOPK] = w
    (HERE / "xm.bin").write_bytes(xm.tobytes())
    (HERE / "w.bin").write_bytes(W.tobytes())
    (HERE / "ref_out.bin").write_bytes(ref.tobytes())
    bufs = [f"buf xm {xm.nbytes} xm.bin", f"buf w {W.nbytes} w.bin"]
    runs = []
    if BIAS:
        b = np.zeros(1024, np.float32)
        b[:E] = bias
        (HERE / "bias.bin").write_bytes(b.tobytes())
        bufs.append("buf bias 4096 bias.bin")
        run_line = "run k xm w bias out"
    else:
        run_line = "run k xm w out"
    bufs.append("buf out 4096")
    cfg = "\n".join([
        "device",
        "xclbin G build/final.xclbin",
        "kernelx k G build/insts.bin",
        *bufs,
        run_line,
        run_line,
        "dump out y_out.bin 4096",
        "",
    ])
    (HERE / "run.cfg").write_text(cfg, newline="\n")
    print(f"top{TOPK}={top.tolist()} w={np.round(w, 4).tolist()} p.max={p.max():.4f} E={E} sigmoid={SIGMOID} bias={BIAS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
