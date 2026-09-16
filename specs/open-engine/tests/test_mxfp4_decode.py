"""OPEN-QUANT-MXFP4: the container's MXFP4 chunk decodes to a defined value array, and the
integer form the AIE tile uses agrees with it.

Runs against a checked-in byte fixture cut from the shipped GPT-OSS-20B-NPU2 container -
eight real chunks spanning gate, up and down slabs at four column blocks - so it needs
neither the 14.4 GB container nor the network.

The discrimination is the point. A reader and a replica that share a dequantiser agree
perfectly while both are wrong, so every positive check here is paired with a near-miss that
must fail.
"""
# Traces: OPEN-QUANT-MXFP4
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "open_kernels"))

from model import mxfp4 as M  # noqa: E402

FIXTURE = HERE / "fixtures" / "gptoss_mxfp4_chunks.npz"


@pytest.fixture(scope="module")
def fx():
    d = np.load(FIXTURE)
    return d["chunks"], d["expect"], d["picks"]


def test_the_ladder_is_twice_the_e2m1_levels():
    """KVALUES are integers because they are doubled, paired with a halved scale. That is
    what lets a GEMV keep its multiply integer, and it is the whole reason the AIE tile can
    reuse the shipped q4_1 kernel's skeleton."""
    assert M.KVALUES.tolist() == [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
    assert np.array_equal(M.KVALUES, M.KVALUES.astype(np.int32).astype(np.float32))


def test_the_integer_decode_equals_the_table():
    """decode_int is the branchless form the tile computes; it must equal the lookup."""
    codes = np.arange(16, dtype=np.uint8)
    assert np.array_equal(M.decode_int(codes).astype(np.float32), M.KVALUES)


@pytest.mark.parametrize("name,f", [
    # no doubling on the top step: 12 becomes 10, 8 becomes 7
    ("no 2x above 6", lambda u: u + np.maximum(u - 4, 0) + np.maximum(u - 6, 0)),
    # the second knee in the wrong place
    ("knee at 5", lambda u: u + np.maximum(u - 4, 0) + 2 * np.maximum(u - 5, 0)),
    # the first knee in the wrong place
    ("knee at 3", lambda u: u + np.maximum(u - 3, 0) + 2 * np.maximum(u - 6, 0)),
    # plain linear, i.e. treating MXFP4 as if it were uniform
    ("linear", lambda u: u),
])
def test_near_miss_ladders_do_not_reproduce_the_table(name, f):
    u = (np.arange(16, dtype=np.uint8) & 0x07).astype(np.int16)
    mag = np.asarray(f(u), np.int16)
    got = np.where((np.arange(16) & 0x08) != 0, -mag, mag).astype(np.float32)
    assert not np.array_equal(got, M.KVALUES), f"{name} must not reproduce KVALUES"


def test_the_sign_bit_is_0x08(fx):
    """Bit 3, not bit 4. Getting it wrong flips half the weights and still looks plausible."""
    chunks, _, _ = fx
    codes = M.chunk_codes(chunks[0])
    wrong = np.where((codes & 0x10) != 0, -1, 1) * np.abs(M.decode_int(codes))
    assert not np.array_equal(wrong, M.decode_int(codes))


def test_decode_reproduces_the_fixture(fx):
    """Every value of eight real chunks, exactly."""
    chunks, expect, _ = fx
    for i in range(len(chunks)):
        got = M.decode_chunk(chunks[i])
        assert got.shape == (32, 128)
        assert np.array_equal(got, expect[i]), f"chunk {i} differs"


def test_the_tile_formula_decodes_the_fixture_too(fx):
    """decode_int is what the AIE tile computes; decode_chunk uses the table. Check the
    FORMULA against real container bytes, not only against the table, or a wrong formula
    only ever meets a synthetic 16-element input."""
    chunks, expect, _ = fx
    for i in range(len(chunks)):
        codes = M.chunk_codes(chunks[i])
        scale = M.e8m0(M.chunk_scales(chunks[i]))
        got = M.decode_int(codes).astype(np.float32) * np.repeat(scale, M.GROUP, axis=1)
        assert np.array_equal(got, expect[i]), f"chunk {i} differs under the tile formula"


def test_the_fixture_is_not_trivially_zero(fx):
    """A fixture of mostly zeros would pass a broken decoder."""
    _, expect, _ = fx
    assert (expect != 0).mean() > 0.5
    assert len(np.unique(np.abs(expect))) > 8


def test_the_scale_is_a_power_of_two(fx):
    """E8M0 carries an exponent and no mantissa, so every scale is exactly a power of two.
    The GEMV leans on this - it is why the scale costs less than q4_1's bf16 `d`."""
    chunks, _, _ = fx
    s = M.e8m0(M.chunk_scales(chunks[0]))
    m, e = np.frexp(s)
    assert np.all(m == 0.5)
    assert np.all(s > 0)


def test_a_column_block_zero_chunk_carries_the_bias_in_its_padding(fx):
    """The converter writes the expert bias twice: as a named tensor and into byte 128 of
    every column-block-0 chunk. The duplicate is what pins this layout unambiguously, since
    it distinguishes gate from up from down by value."""
    chunks, _, picks = fx
    cb0 = [i for i, (slab, q, f) in enumerate(picks.tolist()) if q == 0]
    assert cb0, "fixture must contain a column-block-0 chunk"
    b = M.chunk_bias(chunks[cb0[0]])
    assert b.shape == (32,)
    assert np.isfinite(b).all()
    assert np.abs(b).max() > 0

    # and a chunk from a later column block has zeros there instead
    other = [i for i, (slab, q, f) in enumerate(picks.tolist()) if q != 0]
    if other:
        assert np.all(M.chunk_bias(chunks[other[0]]) == 0)


def test_codes_and_values_are_not_the_same_test(fx):
    """MXFP4 has two zero encodings. This container's source normalises the sign, so raw
    codes differ from upstream's where values do not -- a test asserting code equality fails
    on a decoder that is exact. Pin that the two notions really are different here."""
    chunks, _, _ = fx
    codes = M.chunk_codes(chunks[0])
    flipped = np.where(codes == 0, np.uint8(8), codes)      # +0 -> -0
    assert not np.array_equal(codes, flipped)
    assert np.array_equal(M.KVALUES[codes], M.KVALUES[flipped])
