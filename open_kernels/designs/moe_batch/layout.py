"""The host-side layouts of moe_batch's x, h and y buffers (moe_batch.h's comment), shared by
make_test.py and compare.py; src/open_qwen36/core.cpp writes and reads the same."""
from __future__ import annotations

import numpy as np

NT = 8


def x_to_dev(x: np.ndarray) -> np.ndarray:
    """x [slots, 8 tokens, K] bf16 -> the A tiles [slots][K/8][8 tokens][8 k]."""
    S, T, K = x.shape
    return np.ascontiguousarray(x.reshape(S, T, K // 8, 8).transpose(0, 2, 1, 3))


def y_from_dev(dev: np.ndarray, rows: int) -> np.ndarray:
    """The C tiles [slots][rows/64][4 groups][2 parities][8 tokens][8 j] -> y [slots, 8 tokens, rows],
    row = 64 band + 16 g + 2 j + p."""
    S = dev.size // (rows * NT)
    d = dev.reshape(S, rows // 64, 4, 2, NT, 8)                        # [s, band, g, p, t, j]
    y = d.transpose(0, 4, 1, 2, 5, 3)                                  # [s, t, band, g, j, p]
    return np.ascontiguousarray(y.reshape(S, NT, rows))


def h_from_dev(dev: np.ndarray, ff: int) -> np.ndarray:
    """The down's A tiles [slots][ff/8][8 tokens][8 k] -> h [slots, 8 tokens, ff]."""
    S = dev.size // (ff * NT)
    return np.ascontiguousarray(dev.reshape(S, ff // 8, NT, 8).transpose(0, 2, 1, 3).reshape(S, NT, ff))
