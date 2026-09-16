# Traces: OPEN-PACK-Q4-0 (canonical spec: specs/open-engine/spec.md)
"""A 5120-byte chunk with no mins is a signed quantiser, and the packer transcodes it.

q4_1 reads w = d * q + min over an unsigned nibble; some containers use the same chunk
for w = d * int4(q), a signed nibble with no min, and write every min as zero. Read the
wrong way every block comes out one-sided at about 2.7x the right spread, and the fp64
replica misreads it identically -- so it agrees with the kernels to the bit and the model
answers with noise. See .claude/plans/qwen2-q4-0-container.md.

The transcode is exact: int4(q) == (q ^ 8) - 8, so flipping bit 3 of every nibble turns
two's complement into offset binary and -8 * d goes in the min slot.
"""
from __future__ import annotations

import numpy as np
import pytest

from recipes.pack import is_signed_q4, q4_0_to_q4_1, q4_chunks_of


def bf16(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def unbf16(u):
    return (np.asarray(u, np.uint16).astype(np.uint32) << 16).view(np.float32)


def chunk(scales, mins, nibbles) -> np.ndarray:
    b = np.zeros(5120, np.uint8)
    b[:512] = np.frombuffer(np.ascontiguousarray(scales, np.uint16).tobytes(), np.uint8)
    b[512:1024] = np.frombuffer(np.ascontiguousarray(mins, np.uint16).tobytes(), np.uint8)
    b[1024:] = nibbles
    return b


class FakeContainer:
    """Only what the packer probes for."""

    def __init__(self, fmt=None):
        self.fmt = fmt

    def chunk_bytes_of(self, name):
        return 5120

    def quant_format_of(self, name):
        return self.fmt


SCALES = bf16(np.linspace(0.001, 0.05, 256).astype(np.float32))
NIBBLES = np.arange(4096, dtype=np.uint8)


def signed_chunk():
    return chunk(SCALES, np.zeros(256, np.uint16), NIBBLES)


def test_all_zero_mins_reads_as_the_signed_quantiser():
    c = signed_chunk()
    assert is_signed_q4(FakeContainer(), "w", c.reshape(1, 5120))


def test_one_non_zero_min_is_a_plain_q4_1_tensor():
    mins = np.zeros(256, np.uint16)
    mins[137] = 0xBC00                                   # -1.0
    c = chunk(SCALES, mins, NIBBLES)
    assert not is_signed_q4(FakeContainer(), "w", c.reshape(1, 5120))
    out = q4_chunks_of(FakeContainer(), "w", c)
    assert np.array_equal(out[0], c), "a q4_1 tensor is passed through untouched"


def test_a_container_that_declares_its_format_wins_over_the_data():
    """Nothing writes the stamp yet; this is the hook it lands on."""
    c = signed_chunk()
    assert not is_signed_q4(FakeContainer("q4_1"), "w", c.reshape(1, 5120))
    mins = np.zeros(256, np.uint16); mins[0] = 0xBC00
    assert is_signed_q4(FakeContainer("q4_0"), "w", chunk(SCALES, mins, NIBBLES).reshape(1, 5120))


def test_the_transcode_reproduces_the_signed_values_exactly():
    c = signed_chunk()
    out = q4_0_to_q4_1(c.reshape(1, 5120))
    d = unbf16(np.frombuffer(np.ascontiguousarray(out[0, :512]).tobytes(), np.uint16))
    mn = unbf16(np.frombuffer(np.ascontiguousarray(out[0, 512:1024]).tobytes(), np.uint16))
    assert np.array_equal(mn, -8.0 * d), "min = -8 * d, exactly"
    # every nibble: (q ^ 8) - 8 is the signed value it stood for
    for name, lo, hi in (("low", 0x0F, 0), ("high", 0xF0, 4)):
        src = (c[1024:] & lo) >> hi
        got = (out[0, 1024:] & lo) >> hi
        want = np.where(src < 8, src, src.astype(np.int16) - 16)
        assert np.array_equal(got.astype(np.int16) - 8, want), f"{name} nibble"


def test_the_packer_transcodes_a_signed_tensor_on_the_way_into_the_pool():
    c = signed_chunk()
    out = q4_chunks_of(FakeContainer(), "model.layers.0.self_attn.q_proj.weight", c)
    assert out.shape == (1, 5120)
    assert not np.array_equal(out[0], c), "it is not passed through"
    assert np.array_equal(out[0], q4_0_to_q4_1(c.reshape(1, 5120))[0])


class FakeQ8Head:
    """A container whose lm_head is 4-bit, standing in for `Q4NX` at the one method."""

    def __init__(self, chunk_bytes, nch):
        self.cb = chunk_bytes
        self.hidden = 2048
        self.bytes = np.zeros(nch * chunk_bytes, np.uint8)

    def chunk_bytes_of(self, name):
        return self.cb

    def raw(self, name):
        return self.bytes


def test_the_q8_head_reader_refuses_a_four_bit_head():
    """17 q4_1 chunks are exactly 10 q8 chunks, so the reshape succeeds and the reader
    silently answers with the wrong weights unless it checks (OPEN-PACK-Q4-0)."""
    from q4nx import Q4NX
    f = FakeQ8Head(5120, 17)
    with pytest.raises(ValueError) as e:
        Q4NX.lmhead_logits(f, np.zeros(2048, np.float32))
    assert "lm_head.weight" in str(e.value) and "5120" in str(e.value)
