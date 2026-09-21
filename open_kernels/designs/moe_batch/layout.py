"""The host-side layouts of moe_batch's x, h and y buffers (moe_batch.h's comment), shared by
make_test.py and compare.py; src/open_qwen36/core.cpp writes and reads the same."""
from __future__ import annotations

import os

import numpy as np

NT = int(os.environ.get("MB_NT", 8))


def x_to_dev(x: np.ndarray) -> np.ndarray:
    """x [slots, NT tokens, K] bf16 -> the A tiles [slots][K/8][NT tokens][8 k]."""
    # Token-major inside a k-block already IS [NG sub-tiles][8 tokens][8 k] -- the sub-tile
    # axis is just the high bits of the token index -- so widening NT costs nothing here.
    S, T, K = x.shape
    return np.ascontiguousarray(x.reshape(S, T, K // 8, 8).transpose(0, 2, 1, 3))


def y_from_dev(dev: np.ndarray, rows: int) -> np.ndarray:
    """The C tiles [slots][rows/64][4 groups][2 parities][NT tokens][8 j] -> y [slots, NT tokens, rows],
    row = 64 band + 16 g + 2 j + p. The NT tokens are NG mmul sub-tiles of 8, the sub-tile
    inside the parity: block (band, g, p, s) then lane t * 8 + j."""
    S = dev.size // (rows * NT)
    d = dev.reshape(S, rows // 64, 4, 2, NT // 8, 8, 8)                # [s, band, g, p, sub, t, j]
    y = d.transpose(0, 4, 5, 1, 2, 6, 3)                               # [s, sub, t, band, g, j, p]
    return np.ascontiguousarray(y.reshape(S, NT, rows))


def h_from_dev(dev: np.ndarray, ff: int) -> np.ndarray:
    """The down's A tiles [slots][ff/8][NT tokens][8 k] -> h [slots, NT tokens, ff]."""
    S = dev.size // (ff * NT)
    d = dev.reshape(S, ff // 8, NT // 8, 8, 8)                         # [s, kblock, sub, t, k]
    return np.ascontiguousarray(d.transpose(0, 2, 3, 1, 4).reshape(S, NT, ff))
