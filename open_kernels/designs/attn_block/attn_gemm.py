r"""attn_gemm: the bf16 x bf16 whole-array GEMM that carries block attention on the NPU
(prefill step B, design A -- .claude/plans/prefill-step-b-attention.md).

Attention over a 256-token block is two matrix products per kv head with a row softmax
between them, and the softmax stays on the host:

    S = Q_g . K_g^T     A = Q_g [M = heads_per_kv * T, K = hd]   B = K_g^T [hd, L]
    O = P_g . V_g       A = P_g [M, L]                          B = V_g   [L, hd]

Both are the plain bf16 GEMM this repo already ships for the embedding models
(npu_offload/gemm_rtp/gemm_pretiled.py, mlir-aie's whole_array with a pre-tiled B), built
with its runtime loop bounds (rtp=True): M and K reach the cores as parameters, N only
shapes the instruction stream, so every (L) the window grows through is another
instruction stream over ONE xclbin -- one hardware context for all of attention, whatever
the prompt length. The export checks that the xclbins agree.

A is row-major bf16 [M, K]; B is npue.tile_b(b, 64, 32, 8, 8, "k,n") of the row-major
[K, N]; C is fp32 [M, N] row-major.

Build (WSL, ironenv142):
    AG_M=2048 AG_K=256 AG_N=2048 python build_design.py designs/attn_block/attn_gemm.py designs/attn_block/build_s2048
    AG_M=2048 AG_K=2048 AG_N=256 python build_design.py designs/attn_block/attn_gemm.py designs/attn_block/build_pv2048
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "npu_offload" / "gemm_rtp"))
from gemm_pretiled import pretiled_array  # noqa: E402

# the reference's own tile: 2mk + 2kn + 2mn bytes = 40 KB of the 64 KB L1 at bf16 in, f32 out
M_TILE, K_TILE, N_TILE = 64, 64, 32
N_COLS = 8

M = int(os.environ.get("AG_M", 2048))
K = int(os.environ.get("AG_K", 256))
N = int(os.environ.get("AG_N", 2048))
if M % (M_TILE * 4) or K % K_TILE or N % (N_TILE * N_COLS):
    sys.exit(f"attn_gemm: M={M} K={K} N={N} must tile by ({M_TILE * 4}, {K_TILE}, {N_TILE * N_COLS})")

DESIGN = pretiled_array
SPECIALIZE = dict(M=M, K=K, N=N, m=M_TILE, k=K_TILE, n=N_TILE, n_aie_cols=N_COLS,
                  dtype_in_str="bf16", dtype_out_str="f32", rtp=True)
