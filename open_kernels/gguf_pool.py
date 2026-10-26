"""Shim: re-export from the isolated open_kernels.gguf_pool package.

The actual implementation lives in ``open_kernels.gguf_pool`` package;
this file makes ``from gguf_pool import ...`` work for existing importers.
"""

from open_kernels.gguf_pool import (  # noqa: F401
    CH_Q4,
    CH_Q8,
    Q4_0_BYTES,
    Q4_1_BYTES,
    Q8_0_BYTES,
    BLK,
    chunk_geometry,
    random_q4_blocks,
    random_q8_0_blocks,
    unpack_q4,
    unpack_q8_0,
    dequant_q4,
    dequant_q8_0,
    dequant_q4_chunk,
    dequant_q8_chunk,
    dequant_pool,
    dequant_pool_lmhead,
    pack_q4_pool,
    pack_q8_pool,
    pack_q8_pool_lmhead,
    q4_chunk_bytes,
    q8_chunk_bytes,
)

from open_kernels.gguf_scale_width import (  # noqa: F401
    Q8_CH_F32,
    gguf_scale_width,
)