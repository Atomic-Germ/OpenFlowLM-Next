"""GPT-OSS's GGUF tensors -> the pool bytes recipes/gptoss.py's kernels stream (NumPy).

The C++ packer (src/open_qwen36/pools.cpp: gx_q8 / gx_ug / gx_down / gx_lmhead) writes the same
bytes; tests/test_gptoss_pack.py pins both to one reading of the GGUF.

q8 half-tile (designs/gemv_q4/gemv_q8.h), 16 rows x 256 K, 5120 B:
    [0    : 256 )  bf16 scale[kb*16 + r]     (kb = the 32-wide block, r = the row)
    [256  : 4352)  int8 code[k*16 + r]
    [4352 : 4416)  f32 bias[r], k-tile-0 half-tiles only (GEMV_Q8_PAD_BIAS); zero elsewhere
    value = code * scale
A 64-row band is K/64 half-tiles; half-tile c of a band is the rows' part c % 4 and k-tile c / 4.

Sources: Q8_0 (fp16 d, 32 int8) -> code = q, scale = bf16(d) (RNE; the only rounding here);
MXFP4 (e8m0 e, 16 bytes of nibbles; value j is qs[j] & 15, value j+16 is qs[j] >> 4) ->
code = KVALUES[nibble] (twice the e2m1 level), scale = 2^(e-128) (exact in bf16).
Rows and columns past the tensor's are zero codes and zero scales.

mx4 half-tile (designs/gemv_q4/gemv_mx4.h), the experts' native form, 16 rows x 256 K, 2368 B:
    four 64-K groups g of 576 B: [bf16 scale[kb'*16 + r] for blocks 2g + kb'] (64 B), then the
    nibbles of block 2g and of block 2g + 1 (256 B each: four 8-K octets, each 32 B for rows
    0..7 then 32 B for rows 8..15, nibble j of a half -- byte j/2, low nibble first -- being
    k = j/8, row = j%8); then f32 bias[16] at [2304 : 2368) on k-tile 0.
    nibble = the e2m1 code re-signed as a two's-complement int4: v = p, or -p for a negative,
    except -12 (s, p = 7), which takes int4's spare -8 (MX4_CODE); code = MX4_CODE[v & 15] is
    the q8 tile's code exactly, scale = 2^(e-128) as there.

The head: lm_head_q8.h's f32-scale chunk, 32 rows x 256 K, 9216 B -- f32 scale[kb*32 + r] at
[0:1024), int8 code[(r/16)*4096 + k*16 + r%16] at [1024:9216); 128-row bands of 4 quarters x
K/256 k-tiles, chunk c of a band the quarter c % 4 and k-tile c / 4. Q8_0's fp16 scale widens
to f32 exactly.
"""
from __future__ import annotations

import numpy as np

CH = 5120
PAD_BIAS = 4352
LM_TILE = 9216
MX_CH = 2368
MX_PAD_BIAS = 2304
KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int8)
LEVELS = KVALUES[:8]
# the mx4 nibble's code, indexed by the nibble (v & 15): v 0..7 -> LEVELS, -1..-6 -> -LEVELS[1..6],
# -8 -> -12 (gemv_mx4.h's straight lines), -7 unused
MX4_CODE = np.array([0, 1, 2, 3, 4, 6, 8, 12, -12, 0, -8, -6, -4, -3, -2, -1], np.int8)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bit patterns, round to nearest even (NaN not expected)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_value(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def e8m0_half(e: np.ndarray) -> np.ndarray:
    x = e.astype(np.uint32)
    bits = np.where(x < 2, np.uint32(0x00200000) << x, (x - 1) << np.uint32(23))
    return bits.astype(np.uint32).view(np.float32)


def q8_0_codes(raw: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """GGUF Q8_0 [rows, cols] -> (int8 codes [rows, cols], f32 block scales [rows, cols/32])."""
    b = np.asarray(raw, np.uint8).reshape(rows, cols // 32, 34)
    d = b[:, :, :2].copy().view(np.float16)[..., 0].astype(np.float32)
    return b[:, :, 2:].copy().view(np.int8).reshape(rows, cols), d


def mxfp4_codes(raw: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """GGUF MXFP4 [rows, cols] -> (int8 codes [rows, cols], f32 block scales [rows, cols/32])."""
    b = np.asarray(raw, np.uint8).reshape(rows, cols // 32, 17)
    qs = b[:, :, 1:]
    codes = np.concatenate([KVALUES[qs & 15], KVALUES[qs >> 4]], axis=2).reshape(rows, cols)
    return codes, e8m0_half(b[:, :, 0])


def mxfp4_nibbles(raw: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """GGUF MXFP4 [rows, cols] -> (int8 mx4 nibble values v [rows, cols], f32 block scales [rows, cols/32])."""
    b = np.asarray(raw, np.uint8).reshape(rows, cols // 32, 17)
    qs = b[:, :, 1:]
    n = np.concatenate([qs & 15, qs >> 4], axis=2).reshape(rows, cols)
    p = (n & 7).astype(np.int8)
    neg = np.where(p == 7, np.int8(-8), -p)
    return np.where(n >= 8, neg, p).astype(np.int8), e8m0_half(b[:, :, 0])


def mx4_tiles(v: np.ndarray, scales: np.ndarray, rows_pad: int, cols_pad: int,
              bias: np.ndarray | None = None) -> np.ndarray:
    """re-signed nibbles / block scales (+ per-row bias) -> mx4 half-tiles in band order,
    uint8 [bands, K/64, 2368] (the q8 tiles' order)."""
    q, s = _pad(v, scales, rows_pad, cols_pad)
    nb, nkt = rows_pad // 64, cols_pad // 256
    t = np.zeros((nb, nkt, 4, MX_CH), np.uint8)
    # nibbles: [b, part, h, rr, kt, kb, oc, kk] -> [b, kt, part, kb, oc, h, kk, rr]
    n = (q.reshape(nb, 4, 2, 8, nkt, 8, 4, 8).transpose(0, 4, 1, 5, 6, 2, 7, 3).astype(np.uint8) & 15)
    n = n.reshape(nb, nkt, 4, 4, 2, 256, 2)                                    # [.., g, kb', byte, lo|hi]
    codes = n[..., 0] | (n[..., 1] << 4)
    sb = bf16_bits(s).reshape(nb, 4, 16, nkt, 4, 2).transpose(0, 3, 1, 4, 5, 2)  # [b, kt, part, g, kb', r]
    grp = t[..., :MX_PAD_BIAS].reshape(nb, nkt, 4, 4, 576)
    grp[..., :64] = np.ascontiguousarray(sb).view(np.uint8).reshape(nb, nkt, 4, 4, 64)
    grp[..., 64:] = codes.reshape(nb, nkt, 4, 4, 512)
    t[..., :MX_PAD_BIAS] = grp.reshape(nb, nkt, 4, MX_PAD_BIAS)
    if bias is not None:
        bp = np.zeros(rows_pad, np.float32)
        bp[:len(bias)] = bias
        t[:, 0, :, MX_PAD_BIAS:] = bp.reshape(nb, 4, 16).view(np.uint8).reshape(nb, 4, 64)
    return t.reshape(nb, nkt * 4, MX_CH)


def dequant_mx4_tiles(t: np.ndarray, rows_pad: int, cols_pad: int) -> tuple[np.ndarray, np.ndarray]:
    """mx4 half-tiles in band order -> (f32 [rows_pad, cols_pad] weights as the kernel sees them, f32 bias)."""
    nb, nkt = rows_pad // 64, cols_pad // 256
    t = np.asarray(t, np.uint8).reshape(nb, nkt, 4, MX_CH)
    grp = t[..., :MX_PAD_BIAS].reshape(nb, nkt, 4, 4, 576)
    s = bf16_value(np.ascontiguousarray(grp[..., :64]).view(np.uint16).reshape(nb, nkt, 4, 4, 2, 16))
    b = grp[..., 64:].reshape(nb, nkt, 4, 4, 2, 4, 2, 32)                     # [.., g, kb', oc, h, byte]
    n = np.stack([b & 15, b >> 4], axis=-1).reshape(nb, nkt, 4, 4, 2, 4, 2, 8, 8).astype(np.int8)
    code = MX4_CODE[n].astype(np.float32)                                     # [.., g, kb', oc, h, kk, rr]
    w = code * s.reshape(nb, nkt, 4, 4, 2, 1, 2, 1, 8)                        # row r = 8 h + rr
    # [b, kt, part, g, kb', oc, h, kk, rr] -> [b, part, h, rr, kt, g, kb', oc, kk]
    w = w.transpose(0, 2, 6, 8, 1, 3, 4, 5, 7).reshape(rows_pad, cols_pad)
    bias = np.ascontiguousarray(t[:, 0, :, MX_PAD_BIAS:]).view(np.float32).reshape(rows_pad)
    return w, bias


def mx4_expert_ug(up: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """expert_ug for mx4 half-tiles."""
    nb, per, _ = up.shape
    return np.stack([up, gate], axis=1).reshape(nb * 2 * per * MX_CH)


def _pad(codes: np.ndarray, scales: np.ndarray, rows_pad: int, cols_pad: int):
    r, c = codes.shape
    if r > rows_pad or c > cols_pad or rows_pad % 64 or cols_pad % 256:
        raise ValueError(f"[{r}, {c}] does not pad to [{rows_pad}, {cols_pad}]")
    q = np.zeros((rows_pad, cols_pad), np.int8)
    q[:r, :c] = codes
    s = np.zeros((rows_pad, cols_pad // 32), np.float32)
    s[:r, :c // 32] = scales
    return q, s


def q8_tiles(codes: np.ndarray, scales: np.ndarray, rows_pad: int, cols_pad: int,
             bias: np.ndarray | None = None) -> np.ndarray:
    """codes / block scales (+ per-row bias) -> half-tiles in band order, uint8 [bands, K/64, 5120]."""
    q, s = _pad(codes, scales, rows_pad, cols_pad)
    nb, nkt = rows_pad // 64, cols_pad // 256
    t = np.zeros((nb, nkt, 4, CH), np.uint8)
    sb = bf16_bits(s).reshape(nb, 4, 16, nkt, 8).transpose(0, 3, 1, 4, 2)        # [b, kt, part, kb, r]
    t[..., 0:256] = np.ascontiguousarray(sb).view(np.uint8).reshape(nb, nkt, 4, 256)
    qc = q.reshape(nb, 4, 16, nkt, 256).transpose(0, 3, 1, 4, 2)                  # [b, kt, part, k, r]
    t[..., 256:PAD_BIAS] = np.ascontiguousarray(qc).view(np.uint8).reshape(nb, nkt, 4, 4096)
    if bias is not None:
        bp = np.zeros(rows_pad, np.float32)
        bp[:len(bias)] = bias
        t[:, 0, :, PAD_BIAS:PAD_BIAS + 64] = bp.reshape(nb, 4, 16).view(np.uint8).reshape(nb, 4, 64)
    return t.reshape(nb, nkt * 4, CH)


def dequant_q8_tiles(t: np.ndarray, rows_pad: int, cols_pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-tiles in band order -> (f32 [rows_pad, cols_pad] weights as the kernel sees them, f32 bias)."""
    nb, nkt = rows_pad // 64, cols_pad // 256
    t = np.asarray(t, np.uint8).reshape(nb, nkt, 4, CH)
    s = bf16_value(np.ascontiguousarray(t[..., 0:256]).view(np.uint16).reshape(nb, nkt, 4, 8, 16))
    q = np.ascontiguousarray(t[..., 256:PAD_BIAS]).view(np.int8).reshape(nb, nkt, 4, 256, 16).astype(np.float32)
    w = q * np.repeat(s, 32, axis=3)                                              # [b, kt, part, k, r]
    w = w.transpose(0, 2, 4, 1, 3).reshape(rows_pad, cols_pad)
    bias = np.ascontiguousarray(t[:, 0, :, PAD_BIAS:PAD_BIAS + 64]).view(np.float32).reshape(rows_pad)
    return w, bias


def expert_ug(up: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """One expert's [band][up band | gate band], the order gx_moe's main cores stream it in."""
    nb, per, _ = up.shape
    return np.stack([up, gate], axis=1).reshape(nb * 2 * per * CH)


def lmhead_chunks(codes: np.ndarray, scales: np.ndarray, rows_pad: int, cols_pad: int) -> np.ndarray:
    """codes / scales -> the head's 9216 B chunks in 128-row band order, uint8 [n * 9216]."""
    q, s = _pad(codes, scales, rows_pad, cols_pad)
    if rows_pad % 128:
        raise ValueError("the head's bands are 128 rows")
    nB, nkt = rows_pad // 128, cols_pad // 256
    t = np.zeros((nB, nkt, 4, LM_TILE), np.uint8)
    sb = s.reshape(nB, 4, 32, nkt, 8).transpose(0, 3, 1, 4, 2)                     # [B, kt, q, kb, r]
    t[..., 0:1024] = np.ascontiguousarray(sb).view(np.uint8).reshape(nB, nkt, 4, 1024)
    qc = q.reshape(nB, 4, 2, 16, nkt, 256).transpose(0, 4, 1, 2, 5, 3)            # [B, kt, q, rb, k, r]
    t[..., 1024:] = np.ascontiguousarray(qc).view(np.uint8).reshape(nB, nkt, 4, 8192)
    return t.reshape(-1)


def dequant_lmhead(t: np.ndarray, rows_pad: int, cols_pad: int) -> np.ndarray:
    nB, nkt = rows_pad // 128, cols_pad // 256
    t = np.asarray(t, np.uint8).reshape(nB, nkt, 4, LM_TILE)
    s = np.ascontiguousarray(t[..., 0:1024]).view(np.float32).reshape(nB, nkt, 4, 8, 32)
    q = np.ascontiguousarray(t[..., 1024:]).view(np.int8).reshape(nB, nkt, 4, 2, 256, 16).astype(np.float32)
    q = q.transpose(0, 1, 2, 3, 5, 4).reshape(nB, nkt, 4, 32, 256)                # [B, kt, q, r, k]
    w = q * np.repeat(s.transpose(0, 1, 2, 4, 3), 32, axis=4)
    return w.transpose(0, 2, 3, 1, 4).reshape(rows_pad, cols_pad)
