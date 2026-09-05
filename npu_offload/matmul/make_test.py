"""Test vectors for one whole-array bf16 -> f32 matmul xclbin (the embedding
engine's NPU shapes): random A[M,K], B[K,N] bf16, C = A @ B in fp64.

    python make_test.py -M 512 -K 768 -N 768 [--asset-dir DIR] [--runs N]

Writes a.bin, b.bin, ref_<tag>.bin and run_<tag>.cfg next to this file, with the
xclbin/insts taken from --asset-dir (default: the shipped
src/xclbins/Embedding-Gemma-300M-OpenNPU2/npu_matmul_f32) as m{M}_{K}x{N}.*.
Run with open_kernels/harness/run_kernel, then `python compare.py <tag>`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
SHIPPED = HERE.parent.parent / "src" / "xclbins" / "Embedding-Gemma-300M-OpenNPU2" / "npu_matmul_f32"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=512)
    ap.add_argument("-K", type=int, default=768)
    ap.add_argument("-N", type=int, default=768)
    ap.add_argument("--asset-dir", default=str(SHIPPED))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    A = (rng.standard_normal((a.M, a.K)) * 0.1).astype(np.float32).astype(bfloat16)
    B = (rng.standard_normal((a.K, a.N)) * 0.1).astype(np.float32).astype(bfloat16)
    C = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    tag = f"m{a.M}_{a.K}x{a.N}"
    # Relative to this directory so the cfg is portable across WSL / Windows paths.
    asset = Path(os.path.relpath(Path(a.asset_dir).resolve(), HERE))
    (HERE / f"a_{tag}.bin").write_bytes(A.tobytes())
    (HERE / f"b_{tag}.bin").write_bytes(B.tobytes())
    (HERE / f"ref_{tag}.bin").write_bytes(C.tobytes())
    cfg = ["device",
           f"xclbin G {(asset / (tag + '.xclbin')).as_posix()}",
           f"kernelx k G {(asset / (tag + '.insts')).as_posix()}",
           f"buf a {A.nbytes} a_{tag}.bin",
           f"buf b {B.nbytes} b_{tag}.bin",
           f"buf c {C.nbytes}"]
    cfg += ["run k a b c"] * a.runs
    cfg += [f"dump c y_{tag}.bin {C.nbytes}", ""]
    (HERE / f"run_{tag}.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"{tag}: assets {asset}  ref absmax={np.abs(C).max():.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
