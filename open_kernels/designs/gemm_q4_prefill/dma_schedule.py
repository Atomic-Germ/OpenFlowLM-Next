"""Bounded, order-preserving DMA submission groups for prefill GEMM."""
from __future__ import annotations


def dma_batches(row_groups: int, token_tiles: int, batch_size: int = 1
                ) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Group consecutive (row group, token tile) visits without padding a tail.

    The shim allocator remains the authority on descriptor capacity. Keep this
    search bounded: an unbounded group can exhaust its 16 descriptors per tile.
    """
    if row_groups <= 0 or token_tiles <= 0:
        raise ValueError('positive row-group and token-tile counts required')
    if batch_size not in (1, 2, 4):
        raise ValueError('DMA batch size must be 1, 2 or 4')
    count = row_groups * token_tiles
    return tuple(tuple(divmod(i, token_tiles) for i in range(first, min(first + batch_size, count)))
                 for first in range(0, count, batch_size))
