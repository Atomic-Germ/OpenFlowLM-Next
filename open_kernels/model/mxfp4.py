"""MXFP4 chunks, as GPT-OSS containers ship them.

MXFP4 is 4-bit FLOAT: sixteen unevenly spaced levels sharing one exponent byte per 32
values, where q4_1 is sixteen evenly spaced levels with a scale and a minimum. The engine's
other quantized forms are all integer, so nothing else here reads this.

A chunk is 2560 bytes covering 32 rows x 128 columns of one expert matrix:

    [   0 : 128 )  128 E8M0 exponent bytes, index g*32 + r
                   (g = the group of 32 columns, 0..3; r = the row, 0..31)
    [ 128 : 512 )  384 pad bytes. In column block 0 only, bytes 128..191 carry 32 bf16
                   output biases for this chunk's rows - the converter writes the bias
                   twice, here and as a named tensor.
    [ 512 :2560 )  2048 nibble bytes, index (rg*64 + c)*16 + i, with rg 0 for rows 0..15
                   and 1 for rows 16..31, c the byte-column 0..63 covering columns 2c and
                   2c+1; i < 8 selects column 2c and i >= 8 column 2c+1, and the low and
                   high nibbles of a byte are two adjacent rows. The converter applies
                   that even/odd split itself so the AIE kernel does not have to.

    value = KVALUES[nibble] * 2^(e8m0 - 128)

KVALUES are TWICE the true e2m1 levels, paired with a halved scale, which is llama.cpp's
convention and is what makes them integers - see `decode_int` and why that matters.

Note MXFP4 has two encodings of zero, +0 and -0. This container's source normalises the
sign, so about 7% of raw codes differ from upstream's while every decoded VALUE agrees.
Compare values, never codes.
"""
from __future__ import annotations

import numpy as np

CHUNK = 2560
ROWS = 32
COLS = 128
GROUP = 32
SCALE_BYTES = 128
NIB_OFF = 512

# twice the e2m1 levels; integers, which is what lets a GEMV keep the multiply integer
KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)


def e8m0(x):
    """2^(b-128) for an E8M0 byte, with llama.cpp's subnormal guard below 2."""
    x = np.asarray(x, np.uint8).astype(np.uint32)
    bits = np.where(x < 2, np.uint32(0x00200000) << x, (x - 1) << np.uint32(23))
    return bits.astype(np.uint32).view(np.float32)


def decode_int(codes):
    """4-bit code -> its signed KVALUE as an integer, branchlessly.

        u = code & 7 ; mag = u + max(u-4, 0) + 2*max(u-6, 0) ; negate on bit 3

    This is the form the AIE tile uses. The ladder is not affine in the code - which is why
    the shipped q4_1 GEMV's trick of feeding raw nibbles to an integer matmul does not carry
    over for free - but it is integer, so the multiply downstream still is.
    See .claude/plans/gptoss-mxfp4-gemv.md.
    """
    q = np.asarray(codes, np.uint8)
    u = (q & 0x07).astype(np.int16)
    mag = u + np.maximum(u - 4, 0) + 2 * np.maximum(u - 6, 0)
    return np.where((q & 0x08) != 0, -mag, mag).astype(np.int8)


def chunk_codes(chunk):
    """(32, 128) raw nibble codes out of one chunk's 2048 nibble bytes."""
    d = np.frombuffer(chunk, np.uint8, count=2048, offset=NIB_OFF).reshape(2, 64, 16)
    out = np.empty((ROWS, COLS), np.uint8)
    for rg in range(2):
        for half in range(2):                      # 0 -> column 2c, 1 -> column 2c+1
            blk = d[rg, :, half * 8:(half + 1) * 8]
            out[rg * 16 + 0:rg * 16 + 16:2, half::2] = (blk & 0x0F).T
            out[rg * 16 + 1:rg * 16 + 16:2, half::2] = (blk >> 4).T
    return out


def chunk_scales(chunk):
    """(32 rows, 4 groups) E8M0 exponent bytes."""
    s = np.frombuffer(chunk, np.uint8, count=SCALE_BYTES)
    return s.reshape(4, ROWS).T.copy()


def chunk_bias(chunk):
    """The 32 bf16 output biases a column-block-0 chunk carries in its padding."""
    u = np.frombuffer(chunk, "<u2", count=32, offset=128).astype(np.uint32) << 16
    return u.view(np.float32)


def decode_chunk(chunk):
    """(32, 128) float32 weights."""
    scale = e8m0(chunk_scales(chunk))
    return KVALUES[chunk_codes(chunk)] * np.repeat(scale, GROUP, axis=1)
