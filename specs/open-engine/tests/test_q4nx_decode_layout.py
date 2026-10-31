"""Keep the CPU oracle's Q4 values exact while making chunk traversal contiguous."""
import numpy as np
import pytest

from q4nx import bf16_to_f32, dq_chunks_q4_1, f32_to_bf16


@pytest.mark.parametrize('count', [0, 1, 3, 33])
@pytest.mark.parametrize('strided', [False, True])
def test_q4_decode_is_chunk_contiguous_and_matches_scalar_band_layout(count, strided):
    rng = np.random.default_rng(702)
    storage = rng.integers(0, 256, (count * 2, 5120), dtype=np.uint8)
    chunks = storage[::2] if strided else storage[:count].copy()
    meta = f32_to_bf16(rng.uniform(-2, 2, (count, 512)).astype(np.float32))
    chunks[:, :1024] = meta.view(np.uint8)
    expected = np.empty((count, 32, 8, 32), dtype=np.float32)
    coeff = bf16_to_f32(meta)
    for c in range(count):
        for row in range(32):
            for block in range(8):
                d, m = coeff[c, block * 32 + row], coeff[c, 256 + block * 32 + row]
                for lane in range(32):
                    # Two 16-row groups, then 8 blocks of 32 lanes x 16 rows.
                    nibble = ((row // 16) * 8 + block) * 32 * 16 + lane * 16 + row % 16
                    packed = int(chunks[c, 1024 + nibble // 2])
                    q = np.float32((packed >> (4 * (nibble % 2))) & 15)
                    expected[c, row, block, lane] = q * d + m
    actual = dq_chunks_q4_1(chunks)
    assert actual.tobytes() == expected.tobytes()
    assert actual.flags.c_contiguous, 'chunk-major traversal must not stride across the entire tensor'
