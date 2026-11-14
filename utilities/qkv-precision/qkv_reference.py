"""Diagnostic model of Q4 table and epilogue arithmetic, not an acceptance oracle.

Primary acceptance uses the independent unrounded FP64 matrix product. This
model must be checked against actual NPU captures before attributing an error.
Only finite, normal-range FP32 inputs and BF16 Q4_1 metadata are supported.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'open_kernels/model'))
from q4nx import bf16_to_f32, f32_to_bf16


def split(x):
    x = np.asarray(x, np.float32)
    hi = bf16_to_f32(f32_to_bf16(x))
    lo = bf16_to_f32(f32_to_bf16(x-hi))
    return hi, lo


def activation_table(x, sum_mode='original'):
    x = np.asarray(x, np.float32)
    if (x.ndim != 1 or not x.size or x.size % 256 or not np.isfinite(x).all()
            or sum_mode not in ('original', 'quantized')):
        raise ValueError('invalid activation shape, values or sum mode')
    blocks = x.reshape(-1, 32)
    bits = np.ascontiguousarray(abs(blocks)).view(np.uint32)
    shift = np.minimum(126, 141-(bits.max(axis=1) >> 23).astype(np.int32))
    xi = np.clip(np.rint(np.ldexp(blocks.astype(np.float64), shift[:, None])),
                 -32768, 32767).astype(np.int16)
    reconstructed = np.ldexp(xi.astype(np.float32), -shift[:, None])
    v = blocks if sum_mode == 'original' else reconstructed
    # Match the producer's 32->16->8->4->2->1 FP32 tree.
    v = (v[:, :16]+v[:, 16:]).astype(np.float32)
    v = (v[:, :8]+v[:, 8:]).astype(np.float32)
    for half in (4, 2, 1):
        v = (v[:, :half]+v[:, half:]).astype(np.float32)
    hi, lo = split(v[:, 0])
    return dict(xi=xi, shift=shift, hi=hi, lo=lo, reconstructed=reconstructed)


def unpack_chunks(chunks, k):
    chunks = np.asarray(chunks)
    if (chunks.dtype != np.uint8 or chunks.ndim != 2 or chunks.shape[1] != 5120
            or k <= 0 or k % 256 or not len(chunks) or len(chunks) % (k//256)):
        raise ValueError('invalid Q4_1 raster chunks or width')
    nr, nc = len(chunks)//(k//256), k//256
    meta = bf16_to_f32(np.ascontiguousarray(chunks[:, :1024]).view('<u2'))
    if not np.isfinite(meta).all():
        raise ValueError('non-finite Q4 metadata')
    nibbles = np.empty((len(chunks), 8192), np.uint8)
    nibbles[:, ::2] = chunks[:, 1024:] & 15
    nibbles[:, 1::2] = chunks[:, 1024:] >> 4
    q = nibbles.reshape(nr, nc, 2, 8, 32, 16).transpose(0, 2, 5, 1, 3, 4)
    d, m = (meta[:, start:start+256].reshape(nr, nc, 8, 32)
            .transpose(0, 3, 1, 2).reshape(nr*32, k//32) for start in (0, 256))
    return q.reshape(nr*32, k//32, 32), d, m


def project(chunks, x, sum_mode='original'):
    table = activation_table(x, sum_mode)
    q, d, m = unpack_chunks(chunks, len(x))
    acc = np.zeros(len(q), np.float32)
    for b, sh in enumerate(table['shift']):
        dot = q[:, b].astype(np.int32) @ table['xi'][b].astype(np.int32)
        hi, lo = split(np.ldexp(dot.astype(np.float32), -sh))
        # Products of BF16 operands are exact in FP32; FP64 addition followed
        # by one FP32 rounding models each individual accumulator MAC.
        for left, right in ((hi, d[:, b]), (lo, d[:, b]),
                            (m[:, b], table['hi'][b]), (m[:, b], table['lo'][b])):
            acc = (acc.astype(np.float64)+left.astype(np.float64)*right).astype(np.float32)
    return acc
