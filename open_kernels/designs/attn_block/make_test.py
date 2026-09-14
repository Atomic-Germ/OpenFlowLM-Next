r"""Test vectors for attn_gemm at the 35B's attention shapes: one kv head's group of
8 query heads over a 256-token block (M = 2048 rows of head dim 256) against a window of
L cached rows, as the two GEMMs the host splits attention into.

    python make_test.py [--L 2048] [--seed S]

Writes, for the score GEMM S = Q . K^T (K = 256, N = L):
    a_s<L>.bin   Q, bf16 [2048, 256] row-major
    b_s<L>.bin   K^T pre-tiled (npue.tile_b of the [256, L] operand)
    ref_s<L>.bin fp64 product as f32 [2048, L]
and for the value GEMM O = P . V (K = L, N = 256):
    a_pv<L>.bin  P, bf16 [2048, L] -- rows that sum to 1, the causal tail zero
    b_pv<L>.bin  V pre-tiled ([L, 256])
    ref_pv<L>.bin
plus run_s<L>.cfg / run_pv<L>.cfg for the harness. compare.py <tag> checks c_<tag>.bin.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "npu_offload" / "gemm_rtp"))
from npue import tile_b  # noqa: E402

HPK, T, HD = 8, 256, 256          # query heads per kv head, block, head dim
M = HPK * T
TK, TN, MAC = 64, 32, 8


def bf(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32).astype(bfloat16)


def write_pair(tag: str, a: np.ndarray, b_kn: np.ndarray, out_dir: Path) -> None:
    """a [M, K] bf16 row-major, b_kn [K, N] bf16 row-major; the kernel's B is the tiled form."""
    Kd, Nd = b_kn.shape
    bt = tile_b(b_kn, TK, TN, MAC, MAC, "k,n")
    ref = a.astype(np.float64) @ b_kn.astype(np.float64)
    a.tofile(out_dir / f"a_{tag}.bin")
    np.ascontiguousarray(bt).astype(bfloat16).tofile(out_dir / f"b_{tag}.bin")
    ref.astype(np.float32).tofile(out_dir / f"ref_{tag}.bin")
    cfg = "\n".join([
        "device",
        f"xclbin G build_{tag}/final.xclbin",
        f"kernelx k G build_{tag}/insts.bin",
        f"buf a {a.size * 2} a_{tag}.bin",
        f"buf b {bt.size * 2} b_{tag}.bin",
        f"buf c {M * Nd * 4}",
        "run k a b c", "run k a b c", "run k a b c",
        f"dump c c_{tag}.bin {M * Nd * 4}", "",
    ])
    (out_dir / f"run_{tag}.cfg").write_text(cfg)
    print(f"{tag}: A [{M},{Kd}] B [{Kd},{Nd}] -> C [{M},{Nd}], {2 * M * Kd * Nd / 1e9:.2f} GFLOP")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2048, help="cached rows (a multiple of 256)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    L = a.L
    if L % 256:
        sys.exit("make_test: L must be a multiple of 256 (N tiles by 32 x 8 columns)")
    rng = np.random.default_rng(a.seed)

    # scores: q and k at the scale a normed, roped head has (unit rms), 1/sqrt(hd) folded into q
    q = bf(rng.standard_normal((M, HD)) / np.sqrt(HD))
    k = bf(rng.standard_normal((L, HD)))
    write_pair(f"s{L}", q, np.ascontiguousarray(k.T), HERE)

    # values: P as a real row softmax over a causal window (query row r attends rows <= its position)
    logits = rng.standard_normal((M, L)) * 2.0
    pos = (np.arange(M) % T) + (L - T)                     # the block sits at the window's end
    logits[np.arange(L)[None, :] > pos[:, None]] = -np.inf
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    v = bf(rng.standard_normal((L, HD)))
    write_pair(f"pv{L}", bf(p), v, HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
