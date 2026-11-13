r"""Test vectors for fa_wide (run_kernel + compare.py), OPEN-PREFILL-ATTN-WIDE.

    python make_test.py --xclbin <x> --insts <i> --out <dir> --D 256 --heads 8 --kv-heads 4 --pos0 300 --window 1024
        [--L 256] [--scale 0] [--qk-scale 1] [--runs 3]
    open_kernels\harness\out\run_kernel.exe <dir>\run.cfg
    python compare.py <dir>

L query rows at cache rows pos0 .. pos0 + L - 1 over a [rows, K | V] cache (the engine's decode layout); run_kernel's
`fwpos` points the stream at pos0 and the window first. Q / K are N(0, qk_scale), V N(0, 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--pos0", type=int, default=0)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.0, help="score scale (0: 1 / sqrt(D)); the build's FW_SCALE")
    ap.add_argument("--qk-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    D, L, H, KVH, p0 = a.D, a.L, a.heads, a.kv_heads, a.pos0
    kvw = KVH * D
    rows = p0 + L + 64
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    q = (rng.standard_normal((L, H * D), dtype=np.float32) * a.qk_scale).astype(bfloat16)
    kv = np.zeros((rows, 2 * kvw), dtype=bfloat16)
    kv[:p0 + L, :kvw] = (rng.standard_normal((p0 + L, kvw), dtype=np.float32) * a.qk_scale).astype(bfloat16)
    kv[:p0 + L, kvw:] = rng.standard_normal((p0 + L, kvw), dtype=np.float32).astype(bfloat16)
    q.tofile(out / "q.bin")
    kv.tofile(out / "kv.bin")
    np.savez(out / "ref.npz", q=q.view(np.uint16), kv=kv.view(np.uint16), D=D, L=L, heads=H, kv_heads=KVH, pos0=p0,
             window=a.window, scale=a.scale or D ** -0.5)
    passes = L // (256 if D == 256 else 128)
    ob = L * H * D * 2
    cfg = ["device", f"xclbin G {Path(a.xclbin).resolve()}", f"kernelx k G {Path(a.insts).resolve()}",
           f"buf q {q.nbytes} q.bin", f"buf kv {kv.nbytes} kv.bin", f"buf o {ob}",
           f"fwpos k {p0} {a.window} {D} {2 * kvw} {passes}"]
    cfg += ["run k q kv kv o"] * a.runs + [f"dump o o.bin {ob}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    print(f"fa_wide D={D} L={L} heads={H}/{KVH} pos0={p0} window={a.window} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
