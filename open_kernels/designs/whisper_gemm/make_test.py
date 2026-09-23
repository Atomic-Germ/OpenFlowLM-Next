r"""Test vectors for the whisper_gemm kernel set: one random (A, B) pair per stream with an
fp64 reference, and a harness .cfg that runs the stream from the EXPORTED set (the single
final.xclbin plus that stream's insts_<name>.bin), so the test covers what the engine loads.

    python make_test.py --set DIR [--out DIR] [--seed S]
    ../../harness/out/run_kernel.exe <out>/run_<name>.cfg && python compare.py <out> <name>

A is bf16 [M, K] row-major at unit scale; B is bf16 [K, N] at 1/sqrt(K), the scale of a
trained projection, tiled with npue.tile_b(b, 64, 32, 8, 8, "k,n"); C is fp32 [M, N].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "npu_offload" / "gemm_rtp"))
sys.path.insert(0, str(HERE))
from npue import tile_b  # noqa: E402
import whisper_gemm as wg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, type=Path, help="export_whisper_kernels.py --out DIR")
    ap.add_argument("--out", type=Path, default=HERE / "test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    xclbin = (a.set / "final.xclbin").resolve()
    for name, (M, K, N) in wg.STREAMS.items():
        if a.only and name not in a.only.split(","):
            continue
        A = rng.standard_normal((M, K)).astype(np.float32).astype(bfloat16)
        B = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32).astype(bfloat16)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        bt = np.ascontiguousarray(tile_b(B, wg.K_TILE, wg.N_TILE, 8, 8, "k,n")).astype(bfloat16)
        A.tofile(a.out / f"a_{name}.bin")
        bt.tofile(a.out / f"b_{name}.bin")
        ref.astype(np.float32).tofile(a.out / f"ref_{name}.bin")
        insts = (a.set / f"insts_{name}.bin").resolve()
        cfg = "\n".join([
            "device",
            f"xclbin G {xclbin}",
            f"kernelx k G {insts}",
            f"buf a {M * K * 2} a_{name}.bin",
            f"buf b {K * N * 2} b_{name}.bin",
            f"buf c {M * N * 4}",
            "run k a b c", "run k a b c", "run k a b c",
            f"dump c c_{name}.bin {M * N * 4}", "",
        ])
        (a.out / f"run_{name}.cfg").write_text(cfg)
        print(f"{name}: A [{M},{K}] B [{K},{N}] -> C [{M},{N}], {2 * M * K * N / 1e9:.2f} GFLOP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
