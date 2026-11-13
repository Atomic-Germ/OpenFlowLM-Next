"""recipes/gptoss_pack.py: GGUF Q8_0 / MXFP4 blocks -> the q8 GEMV half-tiles, the mx4 half-tiles (the
experts' native MXFP4 form, designs/gemv_q4/gemv_mx4.h) and the f32-scale head chunks; and
designs/gemm_bfp16/pack.py's MXFP4 B operand (GB_B=mx4, the block prefill's expert GEMM). The runtime's
pools.cpp gx_* / gp_* ops write the same layouts; the NPU check (designs/gptoss/check_layer.py, byte for
byte against the q8 route) and the full-model logits are what tie them to the kernels.
OPEN-FAMILY-GPTOSS, OPEN-QUANT-MXFP4, OPEN-PREFILL-GPTOSS."""
from __future__ import annotations

import numpy as np

from recipes import gptoss_pack as P


def _q8_0(rows, cols, rng):
    b = np.zeros((rows, cols // 32, 34), np.uint8)
    b[:, :, :2] = rng.uniform(1e-3, 1e-1, (rows, cols // 32)).astype(np.float16)[..., None].view(np.uint8)
    b[:, :, 2:] = rng.integers(-127, 128, (rows, cols // 32, 32)).astype(np.int8).view(np.uint8)
    return b.reshape(-1)


def _mxfp4(rows, cols, rng):
    b = np.zeros((rows, cols // 32, 17), np.uint8)
    b[:, :, 0] = rng.integers(115, 137, (rows, cols // 32))
    b[:, :, 1:] = rng.integers(0, 256, (rows, cols // 32, 16))
    return b.reshape(-1)


def _decode_mxfp4(raw, rows, cols):
    b = raw.reshape(rows, cols // 32, 17)
    d = P.e8m0_half(b[:, :, 0])[..., None]
    kv = P.KVALUES.astype(np.float32)
    return (np.concatenate([kv[b[:, :, 1:] & 15], kv[b[:, :, 1:] >> 4]], axis=2) * d).reshape(rows, cols)


def test_mxfp4_half_tiles_are_lossless_and_padding_is_zero():
    rng = np.random.default_rng(0)
    rows, cols = 80, 288                      # neither a whole band nor whole k-tiles
    raw = _mxfp4(rows, cols, rng)
    t = P.q8_tiles(*P.mxfp4_codes(raw, rows, cols), 128, 512)
    w, bias = P.dequant_q8_tiles(t, 128, 512)
    assert np.array_equal(w[:rows, :cols], _decode_mxfp4(raw, rows, cols))
    assert not w[rows:].any() and not w[:, cols:].any() and not bias.any()


def test_q8_0_half_tiles_round_only_the_scale_and_carry_the_bias_in_the_pad():
    rng = np.random.default_rng(1)
    rows, cols = 64, 256
    raw = _q8_0(rows, cols, rng)
    codes, scales = P.q8_0_codes(raw, rows, cols)
    bias = rng.standard_normal(rows).astype(np.float32)
    t = P.q8_tiles(codes, scales, rows, cols, bias)
    w, got = P.dequant_q8_tiles(t, rows, cols)
    want = (codes.reshape(rows, -1, 32) * P.bf16_value(P.bf16_bits(scales))[..., None]).reshape(rows, cols)
    assert np.array_equal(w, want) and np.array_equal(got, bias)
    tiles = t.reshape(1, cols // 256, 4, P.CH)
    assert tiles[0, 0, :, P.PAD_BIAS:P.PAD_BIAS + 64].any()            # k-tile 0 holds it
    assert not tiles[0, 0, :, P.PAD_BIAS + 64:].any()                  # and nothing past it


def test_the_band_law_is_parts_inner_k_tiles_outer():
    """half-tile c of a band covers rows 16 * (c % 4) and k-tile c / 4 (gemv_q8.h)"""
    rows, cols = 64, 512
    codes = (np.arange(rows)[:, None] // 16 * 10 + np.arange(cols)[None, :] // 256).astype(np.int8)
    t = P.q8_tiles(codes, np.ones((rows, cols // 32), np.float32), rows, cols).reshape(-1, P.CH)
    for c in range(8):
        assert t[c, 256] == (c % 4) * 10 + c // 4


def test_expert_ug_interleaves_up_and_gate_per_band():
    up = np.full((3, 2, P.CH), 1, np.uint8)
    gate = np.full((3, 2, P.CH), 2, np.uint8)
    e = P.expert_ug(up, gate).reshape(3, 2, 2, P.CH)
    assert (e[:, 0] == 1).all() and (e[:, 1] == 2).all()


def test_the_head_keeps_q8_0_scales_exact():
    rng = np.random.default_rng(2)
    rows, cols = 130, 320
    raw = _q8_0(rows, cols, rng)
    codes, scales = P.q8_0_codes(raw, rows, cols)
    t = P.lmhead_chunks(codes, scales, 256, 512)
    w = P.dequant_lmhead(t, 256, 512)
    want = (codes.reshape(rows, -1, 32) * scales[..., None]).reshape(rows, cols)
    assert np.array_equal(w[:rows, :cols], want) and not w[rows:].any() and not w[:, cols:].any()


# ---- mx4: the experts as the GGUF's own nibbles (gemv_mx4.h)


def test_the_mx4_nibble_decodes_by_the_kernels_straight_lines():
    """mx4_codes.h: t = v + max(v - 4, 0) - max(-4 - v, 0), code = t + max(t - 8, 0), over every int4 v but
    the unused -7 -- the table the packer encodes against (MX4_CODE) and twice the e2m1 levels"""
    v = np.arange(-8, 8)
    t = v + np.maximum(v - 4, 0) - np.maximum(-4 - v, 0)
    code = t + np.maximum(t - 8, 0)
    for x, c in zip(v, code):
        if x != -7:
            assert P.MX4_CODE[x & 15] == c
    assert sorted(set(P.MX4_CODE.tolist())) == sorted(set(P.KVALUES.tolist()))


def test_mx4_half_tiles_hold_exactly_what_the_q8_tiles_hold_in_less_than_half_the_bytes():
    rng = np.random.default_rng(3)
    rows, cols = 100, 2880                    # GPT-OSS's expert width: neither whole bands nor k-tiles
    raw = _mxfp4(rows, cols, rng)
    bias = rng.standard_normal(rows).astype(np.float32)
    wq, bq = P.dequant_q8_tiles(P.q8_tiles(*P.mxfp4_codes(raw, rows, cols), 128, 3072, bias), 128, 3072)
    t = P.mx4_tiles(*P.mxfp4_nibbles(raw, rows, cols), 128, 3072, bias)
    wm, bm = P.dequant_mx4_tiles(t, 128, 3072)
    assert np.array_equal(wm, wq) and np.array_equal(bm, bq)
    assert np.array_equal(wm[:rows, :cols], _decode_mxfp4(raw, rows, cols))
    assert t.shape == (2, 48, P.MX_CH) and P.MX_CH / P.CH < 0.5


def test_the_mx4_byte_law_is_the_one_gemv_mx4_h_documents():
    """group g = 576 B: 64 B of scales for blocks 2g, 2g + 1, then each block's four octets of 32 B rows 0..7 +
    32 B rows 8..15; nibble j of a half is k = j / 8, row = j % 8, byte j / 2, the low nibble first"""
    rng = np.random.default_rng(4)
    rows, cols = 128, 512
    raw = _mxfp4(rows, cols, rng)
    v, s = P.mxfp4_nibbles(raw, rows, cols)
    t = P.mx4_tiles(v, s, rows, cols)
    for (b, kt, part, kb, oc, h, j) in [(0, 0, 0, 0, 0, 0, 0), (1, 1, 2, 3, 1, 1, 13), (0, 1, 3, 7, 3, 1, 63)]:
        tile = t[b, kt * 4 + part]
        g, kbp = kb // 2, kb % 2
        byte = tile[g * 576 + 64 + kbp * 256 + oc * 64 + h * 32 + j // 2]
        nib = (byte >> 4) if j % 2 else (byte & 15)
        r, k = 64 * b + 16 * part + 8 * h + j % 8, 256 * kt + 32 * kb + 8 * oc + j // 8
        assert nib == (v[r, k] & 15)
        sc = tile[g * 576:g * 576 + 64].view(np.uint16)[kbp * 16 + 8 * h + j % 8]
        assert P.bf16_value(np.array([sc]))[0] == s[r, k // 32]


# ---- the block prefill's expert GEMM B (gemm_bfp16 GB_B=mx4)


def _gemm_pack():
    import importlib.util
    import pathlib
    path = pathlib.Path(P.__file__).resolve().parents[1] / "designs" / "gemm_bfp16" / "pack.py"
    spec = importlib.util.spec_from_file_location("gemm_bfp16_pack", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_gemm_mxfp4_b_operand_is_exact_and_in_pack_bs_tile_order():
    """mx4_b.cc turns a tile into the bfp16 one with q = code, x = e + 5: exact. The tiles sit where pack_b puts
    the same rows' bfp16 tiles (array column c's n-groups c, c + 8, ..., each group's K / 64 tiles)."""
    G = _gemm_pack()
    rng = np.random.default_rng(5)
    N, K = 2048, 2880
    raw = _mxfp4(N, K, rng)
    v, _ = P.mxfp4_nibbles(raw, N, K)
    e = raw.reshape(N, K // 32, 17)[:, :, 0]
    b = G.pack_b_mx4(v, e)
    assert b.size == N // 128 * (K // 64) * G.MX4_TILE
    assert np.array_equal(G.unpack_b_mx4(b, N, K), _decode_mxfp4(raw, N, K))
    # group 9 (array column 1, its second group) is the column's second run of K / 64 tiles
    nk, per_col = K // 64, N // 128 // 8
    tile = b.reshape(-1, G.MX4_TILE)[(1 * per_col + 1) * nk + 3]          # group 9, k-chunk 3
    assert tile[0] == e[9 * 128, 3 * 2] + 5                                # kb32 0, sp 0, half 0, r 0
    assert (tile[256] & 15) == (v[9 * 128, 3 * 64] & 15)                   # sp 0, kb 0, half 0, value 0
