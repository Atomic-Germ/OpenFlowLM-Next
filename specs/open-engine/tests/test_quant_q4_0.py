# Traces: OPEN-PACK-Q4-0 (canonical spec: specs/open-engine/spec.md)
"""A 5120-byte chunk whose mins are all zero is a signed 4-bit quantiser, not q4_1.

Both formats use the same chunk size, so nothing about the container's shape separates
them, and read the wrong way the model answers with noise while agreeing with the fp64
replica to the bit -- the replica misreads it identically. The packer refuses and names
the format rather than packing it. See .claude/plans/qwen2-q4-0-container.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from recipes.pack import q4_chunks_of


class FakeContainer:
    """The two methods the packer probes for."""

    def __init__(self, chunk: np.ndarray):
        self._chunk = chunk

    def chunk_bytes_of(self, name):
        return 5120


def chunk(min_bytes: bytes) -> np.ndarray:
    """One 5120-byte chunk: 256 bf16 scales, 256 bf16 mins, 4096 bytes of nibbles."""
    b = np.zeros(5120, np.uint8)
    b[:512] = np.frombuffer(np.full(256, 0x3C00, np.uint16).tobytes(), np.uint8)   # scales
    b[512:1024] = np.frombuffer(min_bytes, np.uint8)
    b[1024:] = np.arange(4096, dtype=np.uint8)
    return b


def test_a_chunk_with_every_min_zero_is_refused_by_name():
    c = chunk(np.zeros(256, np.uint16).tobytes())
    with pytest.raises(ValueError, match="signed 4-bit quantiser"):
        q4_chunks_of(FakeContainer(c), "model.layers.0.self_attn.q_proj.weight", c)


def test_the_message_names_the_tensor_and_the_transcode():
    c = chunk(np.zeros(256, np.uint16).tobytes())
    with pytest.raises(ValueError) as e:
        q4_chunks_of(FakeContainer(c), "model.layers.7.mlp.down_proj.weight", c)
    msg = str(e.value)
    assert "model.layers.7.mlp.down_proj.weight" in msg
    assert "nibble ^= 8" in msg and "-8 * d" in msg


def test_a_real_q4_1_chunk_still_packs():
    """One non-zero min is enough: the rule is EVERY min zero, which a real tensor does
    not manage even when most of its blocks are symmetric."""
    mins = np.zeros(256, np.uint16)
    mins[137] = 0xBC00                                  # -1.0
    c = chunk(mins.tobytes())
    out = q4_chunks_of(FakeContainer(c), "model.layers.0.mlp.up_proj.weight", c)
    assert out.shape == (1, 5120)
    assert np.array_equal(out[0], c)
