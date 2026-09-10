r"""Test vectors for gemm_q4_prefill: Y[N_WEIGHT,T] = W(q4_1)[N_WEIGHT,K] @ X^T[K,T],
fp64 reference computed from the SAME pool bytes the kernel streams (mirrors
../gemv_q4/make_test.py's discipline exactly, generalised from one activation
vector to T distinct token activations sharing one weight stream).

    python make_test.py --shape qkv --tokens 256 [--runs N] [--seed S]

Shapes (task 0167 stage 14 table):
    qkv       N_WEIGHT=2560 K=2560
    down_proj N_WEIGHT=2560 K=8192
    up_gate   N_WEIGHT=8192 K=2560

Writes w_<tag>.bin (q4_1 pool bytes, shared across all T for one shape/seed),
x_<tag>.bin (pre-tiled bf16 activation), ref_<tag>.bin (fp32[N_WEIGHT*T],
device drain order == row-major [N_WEIGHT,T]) and run_<tag>.cfg.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))                       # open_kernels/ -> q4_1_pack
sys.path.insert(0, str(HERE.parents[2] / "npu_offload" / "gemm_rtp"))  # -> npue.tile_b
from q4_1_pack import pack_q4_1_pool, pool_reference, random_q4_1_blocks  # noqa: E402
from npue import tile_b  # noqa: E402

SHAPES = {  # name: (N_WEIGHT, K)
    "qkv": (2560, 2560),
    "down_proj": (2560, 8192),
    "up_gate": (8192, 2560),
}
M_TILE, K_TILE, TILE_N = 64, 64, 32
MAC_S, MAC_T = 8, 8   # aie2p bf16 mac_dims (4,8,8): s=8, t=8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="qkv", choices=sorted(SHAPES))
    ap.add_argument("--tokens", type=int, default=256, help="T, must be a multiple of tile_n*8")
    ap.add_argument("--tile-n", type=int, default=TILE_N, help="must match GQP_TILE_N the design was built with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3, help="`run` lines in the cfg (timing)")
    a = ap.parse_args()

    n_weight, k = SHAPES[a.shape]
    T = a.tokens
    tile_n = a.tile_n
    if T % (tile_n * 8) != 0:
        print(f"REFUSE: T={T} is not a multiple of tile_n*n_aie_cols={tile_n * 8}")
        return 1
    if n_weight % (M_TILE * 4) != 0 or k % 256 != 0:
        print(f"REFUSE: shape {a.shape} ({n_weight},{k}) does not tile (256-row blocks / 256-K bands)")
        return 1

    tag = f"{a.shape}_t{T}"
    rng = np.random.default_rng(a.seed)

    print(f"packing q4_1 weight [{n_weight},{k}] ...")
    blocks = random_q4_1_blocks(n_weight, k, rng)
    w = pack_q4_1_pool(blocks, rs=2)
    nbytes = len(w)

    print(f"drawing {T} distinct token activations, K={k} ...")
    # T DISTINCT activations (not identical) -- required to catch cross-token
    # / cross-tile mixing bugs, same discipline as ../gemv_q4/make_test.py.
    xs = [rng.standard_normal(k).astype(np.float32).astype(bfloat16) for _ in range(T)]
    assert len({x.tobytes() for x in xs}) == T, "activations collided -- not actually distinct"
    x_tk = np.stack(xs)  # [T, K] token-major, bf16

    print("computing fp64 reference per token from the SAME pool bytes ...")
    refs = np.stack([pool_reference(w, xs[t].astype(np.float32), n_weight, k, rs=2) for t in range(T)])  # [T, N_WEIGHT] f32
    # Device Y is [N_WEIGHT, T] row-major (weight-row-major, matching the
    # kernel's own C tap order) -- transpose the per-token stack to match.
    ref_dev = refs.T.astype(np.float32).copy()  # [N_WEIGHT, T]

    print("pre-tiling activation X^T[K,T] via tile_b ('k,n' order, s=8,t=8) ...")
    x_kt = np.ascontiguousarray(x_tk.T)  # [K, T] -- what the design's B role logically is
    x_tiled = tile_b(x_kt.view(np.uint16), K_TILE, tile_n, MAC_S, MAC_T, order="k,n")
    x_tiled_bf16 = x_tiled.view(bfloat16)

    (HERE / f"w_{a.shape}.bin").write_bytes(w.tobytes())  # weight shared across T for a given (shape, seed)
    (HERE / f"x_{tag}.bin").write_bytes(x_tiled_bf16.tobytes())
    (HERE / f"ref_{tag}.bin").write_bytes(ref_dev.tobytes())

    build = tag
    cfg = ["device",
           f"xclbin G build_{build}/final.xclbin",
           f"kernelx k G build_{build}/insts.bin",
           f"buf w {nbytes} w_{a.shape}.bin",
           f"buf x {x_tiled_bf16.nbytes} x_{tag}.bin",
           f"buf y {ref_dev.nbytes}"]
    cfg += ["run k w x y"] * a.runs
    cfg += [f"dump y y_{tag}.bin {ref_dev.nbytes}", ""]
    (HERE / f"run_{tag}.cfg").write_text("\n".join(cfg), newline="\n")

    print(f"{tag}: N_WEIGHT={n_weight} K={k} T={T} w={nbytes} B x={x_tiled_bf16.nbytes} B "
          f"ref={ref_dev.nbytes} B absmax={np.abs(ref_dev).max():.4g}")
    print(f"build: GQP_N={n_weight} GQP_K={k} GQP_T={T} "
          f"python ../../build_design.py gemm_q4_prefill.py build_{build}")
    print(f"run:   ..\\..\\harness\\out\\run_kernel.exe run_{tag}.cfg && python compare.py {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
