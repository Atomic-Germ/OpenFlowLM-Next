r"""whisper_gemm: the bf16 x bf16 whole-array GEMM that carries Whisper's encoder (issue #72).

Every matrix product of whisper-large-v3-turbo's encoder -- and the one that turns its
output into the four decoder layers' cross-attention K|V -- is a plain GEMM with B a weight
and A an activation, over a 30 s window padded from 1500 frames to M = 1536:

    stream  M x K x N               what
    conv1   3072 x  384 x  1280     im2col of the mel, K tap-major (3 x 128)
    conv2   1536 x 3840 x  1280     im2col, stride 2 (3 x 1280)
    qkv     1536 x 1280 x  3840     Q|K|V fused
    o       1536 x 1280 x  1280
    fc1     1536 x 1280 x  5120
    fc2     1536 x 5120 x  1280
    xkv     1536 x 1280 x 10240     cross K|V of all 4 decoder layers, once per window

That is the GEMM the embedding models already ship (npu_offload/gemm_rtp/gemm_pretiled.py,
mlir-aie's whole_array with a pre-tiled B), with runtime loop bounds (rtp=True) so each
shape is an instruction stream over ONE xclbin -- the attn_block precedent. It is the same
module attn_gemm.py wraps; only the shapes and the two knobs below differ.

    tg_depth 2   two ping-pong halves in flight across a barrier: NpuEmbeddings T61 measured
                 1.034-1.141x of array time, bit-identical, on every model it ships. Depth 3
                 compiles and hangs.
    tile n = 32  hidden 1280 = 5 x 256, so N % (n * 8 columns) holds for every stream;
                 n = 48 does not tile 1280, and n = 64 is over the L1 budget at bf16.

fc1 and xkv have m * 4 * N > 2^20, so the design drains C one row block at a time
(tb_n_rows = 1, gemm_pretiled.py) -- a different DMA program, and the reason the exporter
checks that every stream's xclbin is the same core program.

A is row-major bf16 [M, K]; B is npue.tile_b(b, 64, 32, 8, 8, "k,n") of the row-major
[K, N]; C is fp32 [M, N] row-major.

    WG_M=1536 WG_K=1280 WG_N=3840 python build_design.py designs/whisper_gemm/whisper_gemm.py <out>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent / "npu_offload" / "gemm_rtp"))
from gemm_pretiled import pretiled_array  # noqa: E402

M_TILE, K_TILE, N_TILE = 64, 64, 32
N_COLS = 8
TG_DEPTH = 2

STREAMS = {
    "conv1": (3072, 384, 1280),
    "conv2": (1536, 3840, 1280),
    "qkv": (1536, 1280, 3840),
    "o": (1536, 1280, 1280),
    "fc1": (1536, 1280, 5120),
    "fc2": (1536, 5120, 1280),
    "xkv": (1536, 1280, 10240),
}

M = int(os.environ.get("WG_M", 1536))
K = int(os.environ.get("WG_K", 1280))
N = int(os.environ.get("WG_N", 3840))
if M % (M_TILE * 4) or K % K_TILE or N % (N_TILE * N_COLS):
    sys.exit(f"whisper_gemm: M={M} K={K} N={N} must tile by ({M_TILE * 4}, {K_TILE}, {N_TILE * N_COLS})")

DESIGN = pretiled_array
SPECIALIZE = dict(M=M, K=K, N=N, m=M_TILE, k=K_TILE, n=N_TILE, n_aie_cols=N_COLS,
                  dtype_in_str="bf16", dtype_out_str="f32", rtp=True, tg_depth=TG_DEPTH)
