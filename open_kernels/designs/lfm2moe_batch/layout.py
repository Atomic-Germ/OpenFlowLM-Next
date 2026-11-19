"""Host-side layouts of mbl's x, h and y buffers: gemma4_moe_batch/layout.py (x and h are moe_batch's; y is
[rbg][half][row r][1024 B]) at LFM2.5-24B-A2B's HID 2048 / FF 1536."""
from __future__ import annotations

import os

import numpy as np

NT = int(os.environ.get("MBL_NT", 8))
HID, FF = 2048, 1536


def x_to_dev(x: np.ndarray) -> np.ndarray:
    """x [slots, NT tokens, K] bf16 -> the A tiles [slots][K/8][NT tokens][8 k]."""
    S, T, K = x.shape
    return np.ascontiguousarray(x.reshape(S, T, K // 8, 8).transpose(0, 2, 1, 3))


def y_from_dev(dev: np.ndarray, rows: int = HID) -> np.ndarray:
    """The device y [slots][rbg][half][r][2 groups][2 parities][NG][8 t][8 j] -> y [slots, NT, rows],
    row = 64 band + 16 group + 2 j + parity, band = 4 rbg + r, group = 2 half + (0|1)."""
    S = dev.size // (rows * NT)
    nb = rows // 64
    d = dev.reshape(S, nb // 4, 2, 4, 2, 2, NT // 8, 8, 8)             # [s, rbg, half, r, gg, p, sub, t, j]
    d = d.transpose(0, 1, 3, 2, 4, 5, 6, 7, 8).reshape(S, nb, 4, 2, NT // 8, 8, 8)   # [s, band, g, p, sub, t, j]
    y = d.transpose(0, 4, 5, 1, 2, 6, 3)                               # [s, sub, t, band, g, j, p]
    return np.ascontiguousarray(y.reshape(S, NT, rows))


def h_from_dev(dev: np.ndarray, ff: int = FF) -> np.ndarray:
    """The down's A tiles [slots][ff/8][NT tokens][8 k] -> h [slots, NT tokens, ff]."""
    S = dev.size // (ff * NT)
    d = dev.reshape(S, ff // 8, NT // 8, 8, 8)
    return np.ascontiguousarray(d.transpose(0, 2, 3, 1, 4).reshape(S, NT, ff))
