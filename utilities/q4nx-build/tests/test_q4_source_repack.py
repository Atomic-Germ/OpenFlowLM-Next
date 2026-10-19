"""Cross-format Q4 source repacking must preserve the weights, not re-quant.

gguf.quantize has no Q4_K encoder, so the Q4_K source is hand-built at the
byte level; the checks are about the (scale, min, codes) triples
`GGUFTensor.unpack` now canonicalizes by TARGET format.
"""
import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize, quantize

from q4nx.gguf_tensor import GGUFTensor


def _hand_built_q4_k(blocks=1, seed=0):
    """A valid Q4_K byte blob with known semantics: w = d*s_j*q - dmin*m_j."""
    rng = np.random.default_rng(seed)
    d = np.float16(2 ** -8)
    dmin = np.float16(1.0)
    d_b = np.array([d], dtype=np.float16).view(np.uint8).reshape(1, 2)
    dmin_b = np.array([dmin], dtype=np.float16).view(np.uint8).reshape(1, 2)
    s6 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    m6 = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
    scales = np.empty((blocks, 12), dtype=np.uint8)
    scales[:, 0:4] = s6[0:4]
    scales[:, 4:8] = m6[0:4]
    scales[:, 8:12] = (m6[4:8] << 4) | s6[4:8]
    qs = rng.integers(0, 256, size=(blocks, 128), dtype=np.uint8)
    out = np.concatenate(
        [np.tile(d_b, (blocks, 1)),
         np.tile(dmin_b, (blocks, 1)),
         scales, qs], axis=1)
    return out.astype(np.uint8)


def _q4_k_reference(blocks, rows, cols):
    """The exact values a Q4_K byte blob decodes to."""
    d = blocks[:, 0:2].view(np.float16).astype(np.float32).ravel()
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32).ravel()
    scales = blocks[:, 4:16].astype(np.uint8)
    s6 = np.concatenate([scales[:, 0:4] & np.uint8(0x3F),
                         (scales[:, 8:12] & np.uint8(0xF)) | ((scales[:, 0:4] >> np.uint8(6)) << np.uint8(4))], axis=1)
    m6 = np.concatenate([scales[:, 4:8] & np.uint8(0x3F),
                         (scales[:, 8:12] >> np.uint8(4)) | ((scales[:, 4:8] >> np.uint8(6)) << np.uint8(4))], axis=1)
    qs = blocks[:, 16:144].reshape(-1, 4, 32)
    q = np.stack([qs & 0x0F, qs >> 4], axis=2).reshape(-1, 256).astype(np.float32)
    t = d[:, None] * s6
    u = dmin[:, None] * m6
    w = np.zeros((blocks.shape[0], 256), dtype=np.float32)
    for j in range(8):
        w[:, j * 32:(j + 1) * 32] = t[:, j:j + 1] * q[:, j * 32:(j + 1) * 32] - u[:, j:j + 1]
    return w.reshape(rows, cols)


def _q41_rebuild(d, m, q, cols):
    d16 = d.to(torch.float16).float()
    m16 = m.to(torch.float16).float()
    return (d16.repeat_interleave(32, dim=1) * q + m16.repeat_interleave(32, dim=1)).numpy()


def test_q4k_source_to_q4k_target_is_exact():
    blocks = _hand_built_q4_k(blocks=1)
    t = GGUFTensor("x", (256, 1), blocks.reshape(-1), GGMLQuantizationType.Q4_K)
    ref = _q4_k_reference(blocks, 1, 256)
    d, m, q = t.unpack(GGMLQuantizationType.Q4_K)
    rec = _q41_rebuild(d, -m, q, 256)  # pack_q4k stores u = -m_add
    np.testing.assert_allclose(rec, ref, rtol=0, atol=2 ** -7)


def test_q4k_source_to_q4_1_preserves_codes_and_no_sign_flip():
    blocks = _hand_built_q4_k(blocks=1)
    t = GGUFTensor("x", (256, 1), blocks.reshape(-1), GGMLQuantizationType.Q4_K)
    ref = _q4_k_reference(blocks, 1, 256)
    d, m, q = t.unpack(GGMLQuantizationType.Q4_1)
    rec = _q41_rebuild(d, m, q, 256)
    np.testing.assert_allclose(rec, ref, atol=2 ** -6)
    # The pre-fix path packed +u as the added min and mirrored the weights.
    tt, uu, qq = GGUFTensor.unpack_q4_k(blocks.reshape(-1), 256)
    rec_old = _q41_rebuild(tt, uu, qq, 256)
    assert np.abs(rec_old - ref).max() > 10 * np.abs(rec - ref).max()


def test_q4_1_source_to_q4k_target_negates_min():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((4, 512)).astype(np.float32) * 0.1
    packed = quantize(w, GGMLQuantizationType.Q4_1).copy()
    t = GGUFTensor("x", (512, 4), packed, GGMLQuantizationType.Q4_1)
    ref = dequantize(packed, GGMLQuantizationType.Q4_1)
    d, m, q = t.unpack(GGMLQuantizationType.Q4_K)
    # pack_q4k(d, m, q) stores u = m SUBTRACTED: w = t*q - u must equal d*q + m_add
    rec = _q41_rebuild(d, -m, q, 512)
    np.testing.assert_allclose(rec, ref, atol=2 ** -5)


def test_q4_0_source_to_q4_1_target_folds_offset_into_min():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((4, 512)).astype(np.float32) * 0.1
    packed = quantize(w, GGMLQuantizationType.Q4_0).copy()
    t = GGUFTensor("x", (512, 4), packed, GGMLQuantizationType.Q4_0)
    ref = dequantize(packed, GGMLQuantizationType.Q4_0)
    d, m, q = t.unpack(GGMLQuantizationType.Q4_1)
    rec = _q41_rebuild(d, m, q, 512)
    np.testing.assert_allclose(rec, ref, atol=2 ** -5)
    assert q.min() >= 0 and q.max() <= 15


def test_q4k_source_to_q8_0_target_requantizes():
    blocks = _hand_built_q4_k(blocks=1)
    t = GGUFTensor("x", (256, 1), blocks.reshape(-1), GGMLQuantizationType.Q4_K)
    d, _, qw = t.unpack(GGMLQuantizationType.Q8_0)
    # A Q8_0 requant means a signed int8 code stream, not 4-bit codes.
    assert qw.min() >= -128 and qw.max() <= 127 and qw.dtype != torch.float32
