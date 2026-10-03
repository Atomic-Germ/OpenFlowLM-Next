"""Segment geometry over an unchanged Q4 standard (64-row band) pool."""

SEGMENT_K = 8192


def segments(k: int) -> tuple[tuple[int, int], ...]:
    if k <= 0 or k % 256:
        raise ValueError("segmented K must be positive and 256-aligned")
    return tuple((off, min(SEGMENT_K, k - off)) for off in range(0, k, SEGMENT_K))


def weight_slice(k: int, band: int, start: int, width: int) -> tuple[int, int]:
    """Byte offset/length inside the original matrix, never a repacked segment."""
    if k <= 0 or k % 256 or band < 0 or start < 0 or width <= 0 or start % 256 or width % 256 or start + width > k:
        raise ValueError("invalid segmented Q4 band slice")
    # Each 256 columns contains two consecutive 5120-byte, 32-row chunks.
    return (band * (k // 256) + start // 256) * 10240, width // 256 * 10240
