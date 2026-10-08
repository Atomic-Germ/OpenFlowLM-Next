r"""Host packing for dit_conv's weights: what make_test.py tests is what the model packer ships.

pack_conv(w, b, taps, up) -> uint8, per virtual output-channel tile v (128 channels; with
`up`, v = phase * (Cout/128) + channel tile):
    one bias object   1024 v8bfp16ebs8 elements (9216 bytes): the tile's 128 biases as
                      raw bf16 in the first 256 bytes, zeros after (conv.cc)
    the K walk        dit_gemm's pack_b of the [K, 128] weight slice, K ordered
                      (cin chunk of 64, ky, kx, channel) -- the band readers' order
Cin is zero-padded to a multiple of 64 and Cout to one of 128.

`up` (nearest 2x upsample, then the conv) is computed on the source grid as 4 output
phases (py, px). Output row 2y + py reads upsampled row 2y + py - 1 + ky, which is source
row y - 1 + sy with sy = (py + ky + 1) // 2; so each phase is a 3x3 conv on the source
whose tap (sy, sx) is the sum of the original taps that land on it (4 of its 9 taps are
non-zero).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dit_gemm"))
from pack import pack_b, unpack_b  # noqa: E402

K_T, N_T = 64, 128
B_OBJ = 9216            # bytes of one 64 x 128 bfp16ebs8 B object


def pad_channels(w: np.ndarray, b: np.ndarray, cin: int | None = None):
    """Zero-pad Cout to a multiple of 128 and Cin to a multiple of 64 (or to `cin`, the
    input tensor's channel count, when it has more)."""
    co, ci = w.shape[:2]
    cop, cip = -(-co // N_T) * N_T, max(-(-ci // K_T) * K_T, cin or 0)
    wp = np.zeros((cop, cip) + w.shape[2:], np.float32)
    wp[:co, :ci] = w
    bp = np.zeros(cop, np.float32)
    bp[:co] = b
    return wp, bp


def phase_weights(w: np.ndarray) -> np.ndarray:
    """[Cout, Cin, 3, 3] -> [4 phases (py, px), Cout, Cin, 3 sy, 3 sx]."""
    out = np.zeros((4,) + w.shape, np.float64)
    for py in range(2):
        for px in range(2):
            for ky in range(3):
                for kx in range(3):
                    sy, sx = (py + ky + 1) // 2, (px + kx + 1) // 2
                    out[2 * py + px, :, :, sy, sx] += w[:, :, ky, kx]
    return out


def k_rows(w: np.ndarray) -> np.ndarray:
    """[Cout, Cin, kh, kw] -> [K, Cout], K = (cin chunk, ky, kx, channel in chunk)."""
    co, ci, kh, kw = w.shape
    t = w.reshape(co, ci // K_T, K_T, kh, kw).transpose(1, 3, 4, 2, 0)
    return t.reshape(ci * kh * kw, co)


def pack_conv(w: np.ndarray, b: np.ndarray, taps: int = 9, up: bool = False,
              cin: int | None = None) -> np.ndarray:
    """One conv (w [Cout, Cin, kh, kw], b [Cout]); `up`: nearest-2x upsample first;
    `cin`: the input tensor's channel count if it is padded beyond the weights'."""
    w = np.asarray(w, np.float32)
    assert w.shape[2] * w.shape[3] == taps, (w.shape, taps)
    if up:
        return pack_conv_phases(phase_weights(w), np.stack([np.asarray(b)] * 4), cin)
    return pack_conv_phases(w[None], np.asarray(b)[None], cin)


def pack_conv_phases(wp: np.ndarray, bp: np.ndarray, cin: int | None = None) -> np.ndarray:
    """Per-phase weights wp [phases, Cout, Cin, kh, kw] and biases bp [phases, Cout]:
    virtual tile v = phase * (Cout/128) + channel tile (dit_conv.py's "up")."""
    out = []
    for w, b in zip(wp, bp):
        w, b = pad_channels(np.asarray(w, np.float32), np.asarray(b, np.float32), cin)
        m = k_rows(w)
        for ct in range(m.shape[1] // N_T):
            bias = np.zeros(B_OBJ, np.uint8)
            bias[:2 * N_T] = b[ct * N_T:(ct + 1) * N_T].astype(bfloat16).view(np.uint8)
            out += [bias, pack_b(m[:, ct * N_T:(ct + 1) * N_T])]
    return np.concatenate(out)


def unpack_conv(packed: np.ndarray, cin: int, cout: int, taps: int = 9, up: bool = False):
    """The weights and biases the kernel uses: ([phases, Cout, Cin_pad, kh, kw] float64,
    [phases, Cout_pad] float64 bf16 biases). cin/cout are the padded counts."""
    k = cin * taps
    per = B_OBJ + k * N_T * 9 // 8
    n_ph = 4 if up else 1
    kh = 3 if taps == 9 else 1
    ws = np.zeros((n_ph, cout, cin, kh, kh))
    bias = np.zeros((n_ph, cout))
    for p in range(n_ph):
        for ct in range(cout // N_T):
            blk = packed[(p * (cout // N_T) + ct) * per:][:per]
            bias[p, ct * N_T:(ct + 1) * N_T] = (blk[:2 * N_T].view(np.uint16).astype(np.uint32)
                                                 << 16).view(np.float32)
            m = unpack_b(blk[B_OBJ:], k, N_T)                    # [K, 128]
            t = m.reshape(cin // K_T, kh, kh, K_T, N_T).transpose(4, 0, 3, 1, 2)
            ws[p, ct * N_T:(ct + 1) * N_T] = t.reshape(N_T, cin, kh, kh)
    return ws, bias
