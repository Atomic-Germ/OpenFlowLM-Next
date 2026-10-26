"""GGUF pool packing/unpacking module (isolated).

This package provides the GGUF pool operations that were previously available
as a single module at ``open_kernels.gguf_pool``.  The individual functions
live in ``gguf_pool.py`` and ``gguf_scale_width.py`` within this package.
"""

from .gguf_pool import (  # noqa: F401
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

# Q8_CH_F32 is defined in gguf_scale_width; re-export for convenience
from .gguf_scale_width import Q8_CH_F32  # noqa: F401