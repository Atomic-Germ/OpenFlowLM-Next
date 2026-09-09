# Traces: OPEN-QUANT-Q4K, OPEN-PACK-PLAN (canonical spec: specs/open-engine/spec.md)
"""Q4_K containers: the 4736-byte chunk, its transcode to the pool's q4_1 chunk, and the
agreement between the converter that writes the format and the packers that read it.

The plan is specs/open-engine/plans/q4k-containers.md.

A Q4_K chunk holds the same 32-row x 256-column tile a q4_1 chunk does, and its scale and
min have the same granularity -- one pair per (row, 32-column group) -- so the transcode
is two multiplies per entry plus a byte de-interleave, and no chunk index law, plan,
manifest or kernel moves. What it costs is one bf16 rounding of each product, which the
first test bounds exactly.

  q4k chunk, 4736 B:  scales[8][32] uint8 at [0, 256)    index g*32 + r
                      mins  [8][32] uint8 at [256, 512)  same index
                      qs    [256][16]      at [512, 4608)  byte k*16 + r/2, even row low
                      S     [32]    bf16   at [4608, 4672) index r
                      M     [32]    bf16   at [4672, 4736) index r, stored NEGATED
  value(r, k) = S[r] * scales[k/32][r] * nib + M[r] * mins[k/32][r]

The same synthetic chunks are built, transcoded and hashed by
src/open_qwen36/pools_test.cpp, so the two packers are held to each other byte for byte
without a model file.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recipes import pack

from test_pack_plan import _fnv1a, _lcg_bytes

REPO = Path(__file__).resolve().parents[3]

Q4K = 4736
CH = pack.CH
Q8 = pack.Q8

OUT_DIM, IN_DIM = 128, 512
NCH = (OUT_DIM // 32) * (IN_DIM // 256)        # 8 chunks: 32 rows x 256 K each
Q4K_NAME = "model.layers.{l}.mlp.gate_proj.weight"
Q4_NAME = "model.layers.{l}.mlp.down_proj.weight"
BAD_NAME = "model.layers.{l}.mlp.up_proj.weight"

# FNV-1a 64 of the transcoded chunks of _q4k_vector(NCH). src/open_qwen36/pools_test.cpp
# asserts the same number on the same bytes, so a divergence in either packer fails one
# of the two tests.
Q4K_TRANSCODE_FNV1A = 0x685DC049EC1CA2D7
# ... and of the pool std_perm puts them in, beside a q4_1 tensor.
Q4K_POOL_FNV1A = 0xB02083912551D9D3


def _bf16(u16) -> np.ndarray:
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _q4k_vector(nch: int) -> np.ndarray:
    """The vector pools_test.cpp builds, byte for byte: per chunk one LCG seeded from the
    chunk index; 4608 plain bytes (the uint8 scales, the uint8 mins and the nibbles), then
    32 bf16 `S` with exponent 0x76 (finite, positive) and 32 bf16 `M` with the same
    exponent and the sign bit set -- the converter stores the min already negated, and
    Q4_K's own dmin is non-negative, so a real `M` is never positive."""
    M32 = 0xFFFFFFFF
    out = np.zeros((nch, Q4K), np.uint8)
    for c in range(nch):
        s = (0x9E3779B9 * (c + 1)) & M32
        for i in range(4608):
            s = (s * 1664525 + 1013904223) & M32
            out[c, i] = s >> 24
        sm = np.zeros(64, np.uint16)
        for i in range(32):
            s = (s * 1664525 + 1013904223) & M32
            sm[i] = 0x3B00 | (s >> 24)
        for i in range(32):
            s = (s * 1664525 + 1013904223) & M32
            sm[32 + i] = 0xBB00 | (s >> 24)
        out[c, 4608:] = sm.view(np.uint8)
    return out


def _q4k_nibbles(chunk) -> np.ndarray:
    """[4736] chunk bytes -> [32, 256] uint8 (row, k), read straight off the layout above:
    byte 512 + k*16 + r/2, low nibble for an even row."""
    b = np.asarray(chunk, np.uint8)
    qs = b[512:4608].reshape(256, 16)                       # [k, r // 2]
    r = np.arange(32)
    lo = qs[:, r // 2] & 0xF
    hi = qs[:, r // 2] >> 4
    return np.where(r % 2 == 0, lo, hi).T.astype(np.uint8)  # [r, k]


def _q4_1_nibbles(chunk) -> np.ndarray:
    """[5120] pool chunk bytes -> [32, 256] uint8 (row, k): nibble (r/16)*4096 + k*16 + r%16."""
    b = np.asarray(chunk, np.uint8)
    n = np.empty(8192, np.uint8)
    n[0::2] = b[1024:] & 0xF
    n[1::2] = b[1024:] >> 4
    r = np.arange(32)[:, None]
    k = np.arange(256)[None, :]
    return n[(r // 16) * 4096 + k * 16 + (r % 16)]


def _q4k_exact(chunk) -> np.ndarray:
    """[4736] chunk bytes -> [32, 8, 32] f64 (row, block, lane), the value the container
    says, in f64. Written from the layout table, not from the implementation."""
    b = np.asarray(chunk, np.uint8)
    scales = b[:256].astype(np.float64).reshape(8, 32)                       # [g, r]
    mins = b[256:512].astype(np.float64).reshape(8, 32)                      # [g, r]
    S = _bf16(np.ascontiguousarray(b[4608:4672]).view(np.uint16)).astype(np.float64)
    M = _bf16(np.ascontiguousarray(b[4672:4736]).view(np.uint16)).astype(np.float64)
    nib = _q4k_nibbles(b).astype(np.float64).reshape(32, 8, 32)              # [r, g, i]
    return (S[:, None, None] * scales.T[:, :, None] * nib
            + M[:, None, None] * mins.T[:, :, None])


class Q4kContainer:
    """One Q4_K tensor, one q4_1 tensor and one with 1280-byte chunks -- the shape of a
    container from FLM 1.0.3+, which keeps Q4_K where the closed runtime demands it and
    leaves the rest alone."""

    def __init__(self):
        self.data = {
            Q4K_NAME.replace("{l}", "0"): _q4k_vector(NCH).reshape(-1),
            Q4_NAME.replace("{l}", "0"): _lcg_bytes(0x1234567, NCH * CH),
            BAD_NAME.replace("{l}", "0"): _lcg_bytes(0x89ABCDE, NCH * 1280),
        }
        self.chunks = {Q4K_NAME.replace("{l}", "0"): Q4K,
                       Q4_NAME.replace("{l}", "0"): CH,
                       BAD_NAME.replace("{l}", "0"): 1280}

    def raw(self, name):
        return self.data[name]

    def chunk_bytes_of(self, name):
        return self.chunks[name]


def _q4k_pool() -> np.ndarray:
    m = Q4kContainer()
    dst = np.zeros(2 * NCH * CH, np.uint8)
    pack.apply_op({"op": "std_perm", "tensor": Q4K_NAME, "dst": 0, "nch": NCH, "in_dim": IN_DIM},
                  m, 0, dst)
    pack.apply_op({"op": "std_perm", "tensor": Q4_NAME, "dst": NCH * CH, "nch": NCH, "in_dim": IN_DIM},
                  m, 0, dst)
    return dst


# ------------------------------------------------------------------- the transcode
def test_the_transcode_reads_as_the_container_says():
    """A transcoded chunk read as `nib*d + m` equals what the Q4_K bytes mean, to within
    the bf16 rounding of the two products and no further. That rounding is the whole cost
    of the format collapse: the exact product of a bf16 `S` and a uint8 `scales` needs 16
    significand bits and the pool's `d` holds 8, so each group's scale moves by at most a
    half-ulp of bf16 -- 2^-8 relative."""
    src = _q4k_vector(NCH)
    from q4nx import dq_chunks_q4_1
    got = dq_chunks_q4_1(pack.q4k_to_q4_1(src)).astype(np.float64)

    b = src
    scales = b[:, :256].astype(np.float64).reshape(NCH, 8, 32).transpose(0, 2, 1)[..., None]
    mins = b[:, 256:512].astype(np.float64).reshape(NCH, 8, 32).transpose(0, 2, 1)[..., None]
    S = _bf16(np.ascontiguousarray(b[:, 4608:4672]).view(np.uint16)).astype(np.float64)
    M = _bf16(np.ascontiguousarray(b[:, 4672:4736]).view(np.uint16)).astype(np.float64)
    nib = np.stack([_q4k_nibbles(c) for c in b]).astype(np.float64).reshape(NCH, 32, 8, 32)
    a = S[:, :, None, None] * scales * nib          # the scale term
    c = M[:, :, None, None] * mins                  # the min term
    want = a + c

    tol = (2.0 ** -8 + 2.0 ** -20) * (np.abs(a) + np.abs(c))
    err = np.abs(got - want)
    assert (err <= tol).all(), f"worst {(err / np.maximum(tol, 1e-300)).max():.3f} x the bf16 bound"
    assert err.max() > 0, "the test vector must actually exercise the rounding"


def test_the_nibble_deinterleave_is_exact():
    """Q4_K keeps a column's 32 rows in 16 contiguous bytes; the pool splits rows 0-15 and
    16-31 into two 2048-byte planes. Nothing else about the nibbles changes -- both are
    unsigned uint4 against a (scale, min) pair -- so this is equality, not a tolerance."""
    src = _q4k_vector(NCH)
    out = pack.q4k_to_q4_1(src)
    for c in range(NCH):
        assert np.array_equal(_q4_1_nibbles(out[c]), _q4k_nibbles(src[c])), f"chunk {c}"


def test_the_scale_index_does_not_move():
    """Both formats index their per-(group, row) metadata `g*32 + r`, so the transcode is a
    multiply in place: no permutation of the 256 entries."""
    src = _q4k_vector(1)
    out = pack.q4k_to_q4_1(src)
    d = _bf16(np.ascontiguousarray(out[0, :512]).view(np.uint16)).astype(np.float64)
    m = _bf16(np.ascontiguousarray(out[0, 512:1024]).view(np.uint16)).astype(np.float64)
    S = _bf16(np.ascontiguousarray(src[0, 4608:4672]).view(np.uint16)).astype(np.float64)
    M = _bf16(np.ascontiguousarray(src[0, 4672:4736]).view(np.uint16)).astype(np.float64)
    r = np.arange(256) % 32
    want_d = S[r] * src[0, :256].astype(np.float64)
    want_m = M[r] * src[0, 256:512].astype(np.float64)
    assert (np.abs(d - want_d) <= 2.0 ** -8 * np.abs(want_d)).all()
    assert (np.abs(m - want_m) <= 2.0 ** -8 * np.abs(want_m)).all()
    assert (m <= 0).all(), "a real Q4_K min is stored negated, so the pool's m is never positive"


def test_the_numpy_and_cpp_transcodes_agree():
    """The gate on the two implementations. src/open_qwen36/pools_test.cpp builds the same
    chunks and asserts this same FNV-1a; if either transcode changes, one of the two fails."""
    assert _fnv1a(pack.q4k_to_q4_1(_q4k_vector(NCH)).tobytes()) == Q4K_TRANSCODE_FNV1A, \
        "the Q4_K transcode changed; update src/open_qwen36/pools_test.cpp too"


# ------------------------------------------------------------------- as a pack source
def test_a_q4k_source_packs_in_the_same_chunk_order():
    """The permutation is applied to transcoded chunks, exactly as it is to file chunks: a
    Q4_K chunk and a q4_1 chunk hold the same 32 x 256 tile, so std_perm does not move."""
    got = _q4k_pool()[:NCH * CH].reshape(NCH, CH)
    want = pack.q4k_to_q4_1(_q4k_vector(NCH))[pack.std_perm(NCH, IN_DIM)]
    assert np.array_equal(got, want)


def test_a_q4_1_tensor_beside_it_is_still_a_verbatim_chunk_copy():
    """The q4_1 path must not notice that Q4_K exists: still a chunk copy in std_perm
    order, which is what keeps OPEN-PACK-PLAN's frozen pools byte-equal."""
    m = Q4kContainer()
    got = _q4k_pool()[NCH * CH:].reshape(NCH, CH)
    src = m.raw(Q4_NAME.replace("{l}", "0")).reshape(NCH, CH)
    assert np.array_equal(got, src[pack.std_perm(NCH, IN_DIM)])


def test_the_numpy_and_cpp_packers_agree_on_a_q4k_container():
    assert _fnv1a(_q4k_pool().tobytes()) == Q4K_POOL_FNV1A, \
        "the Q4_K pool changed; update src/open_qwen36/pools_test.cpp too"


def test_q8_perm_refuses_a_q4k_source():
    """A kernel set built to stream a projection at q8 is not satisfied by Q4_K: the pool
    would hold 5120-byte q4_1 half-tiles' worth of the wrong thing. Refused by name and
    both byte counts."""
    m = Q4kContainer()
    with pytest.raises(ValueError) as e:
        pack.q8_chunks_of(m, Q4K_NAME.replace("{l}", "0"), m.raw(Q4K_NAME.replace("{l}", "0")))
    msg = str(e.value)
    assert "gate_proj" in msg and "4736" in msg and "8704" in msg


def test_a_chunk_width_that_is_none_of_the_three_is_still_refused():
    """1280 stays refused, naming the tensor, the count and what the count probably is --
    and the message no longer offers 4736 as an example of something unreadable."""
    m = Q4kContainer()
    with pytest.raises(ValueError) as e:
        pack.q4_chunks_of(m, BAD_NAME.replace("{l}", "0"), m.raw(BAD_NAME.replace("{l}", "0")))
    msg = str(e.value)
    assert "up_proj" in msg and "1280" in msg
    assert "4736" in msg, "the accepted widths should now include Q4_K"
    assert "Q4_K (FLM 1.0.3), which needs a different dequant" not in msg


# ------------------------------------------------------------------- the fp64 reader
def test_dq_chunks_q4_k_reads_the_container_exactly():
    """The reference dequant used by the replica reads the Q4_K bytes themselves, with no
    bf16 collapse -- that is what makes an A/B against the transcode measure the transcode."""
    from q4nx import dq_chunks_q4_k
    src = _q4k_vector(NCH)
    got = dq_chunks_q4_k(src).astype(np.float64)
    want = np.stack([_q4k_exact(c) for c in src])
    assert np.abs(got - want).max() <= 1e-6 * np.abs(want).max()


def test_dq_tile_reads_the_pool_by_default_and_the_container_on_request():
    """`dq_tile` mirrors the packer's decision, as it does for q8: by default a Q4_K tensor
    reads as the transcoded q4_1 the NPU actually holds, so a slice comparison measures the
    KERNELS; `requant=False` reads the container's own Q4_K values, and the gap between the
    two readings is the transcode's quality number."""
    from q4nx import CHUNK_Q4K, Q4NX, dq_chunks_q4_1, dq_chunks_q4_k
    src = _q4k_vector(NCH)
    raw = src.reshape(-1).tobytes()

    me = SimpleNamespace(requant_q8=True)          # the default a real Q4NX carries
    packed = Q4NX.dq_tile(me, raw, OUT_DIM, IN_DIM, chunk=CHUNK_Q4K)
    native = Q4NX.dq_tile(me, raw, OUT_DIM, IN_DIM, chunk=CHUNK_Q4K, requant=False)
    assert packed.shape == (OUT_DIM, IN_DIM) and native.shape == (OUT_DIM, IN_DIM)

    def _raster(v):
        return v.reshape(NCH, 32, 256).reshape(OUT_DIM // 32, IN_DIM // 256, 32, 256) \
                .transpose(0, 2, 1, 3).reshape(OUT_DIM, IN_DIM)

    assert np.allclose(packed, _raster(dq_chunks_q4_1(pack.q4k_to_q4_1(src)).reshape(NCH, 32, 256)))
    assert np.allclose(native, _raster(dq_chunks_q4_k(src).reshape(NCH, 32, 256)))
    assert not np.array_equal(packed, native), "the two readings must actually differ"


# ------------------------------------------------------- the writer and the reader agree
def _converter_packer():
    """`utilities/q4nx-build`'s Q4_K writer, loaded by path: the converter package's name
    (`q4nx`) collides with open_kernels/model/q4nx.py, so it is never put on sys.path."""
    p = REPO / "utilities" / "q4nx-build" / "q4nx" / "gguf_tensor.py"
    spec = importlib.util.spec_from_file_location("q4nx_build_gguf_tensor", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_converters_writer_and_our_reader_agree_on_the_layout():
    """The only cross-check available until a Q4_K container ships: bytes written by
    `q4nx-build` from known (scale, min, quant) triples, read back by our dequant.

    The nibbles come back exactly -- that is the layout check, and a transposed or
    misaligned qs plane could not survive it. The values come back to the re-fit's own
    accuracy: the writer re-expresses each 8-group super-block's exact scales as
    (bf16 S, uint8 s_j), which is where the small residual lives.
    """
    torch = pytest.importorskip("torch")
    gt = _converter_packer()
    rows, cols = 64, 512
    rng = np.random.default_rng(7)
    t = rng.uniform(0.002, 0.02, (rows, cols // 32)).astype(np.float32)       # per-group scale
    u = rng.uniform(0.0, 0.05, (rows, cols // 32)).astype(np.float32)         # per-group min
    q = rng.integers(0, 16, (rows, cols)).astype(np.float32)

    packed = gt.pack_q4k(torch.from_numpy(t), torch.from_numpy(u), torch.from_numpy(q),
                         row_block_size=32, col_block_size=256, keep_block_in_2D=False)
    chunks = packed.contiguous().view(torch.uint8).numpy().reshape(-1, Q4K)
    assert chunks.shape[0] == (rows // 32) * (cols // 256)

    from q4nx import dq_chunks_q4_k
    got = dq_chunks_q4_k(chunks).reshape(-1, 32, 256)
    want = (t[:, :, None] * q.reshape(rows, cols // 32, 32) - u[:, :, None]).reshape(rows, cols)
    want = want.reshape(rows // 32, 32, cols // 256, 256).transpose(0, 2, 1, 3).reshape(-1, 32, 256)

    nib = np.stack([_q4k_nibbles(c) for c in chunks])
    q_chunked = q.reshape(rows // 32, 32, cols // 256, 256).transpose(0, 2, 1, 3).reshape(-1, 32, 256)
    assert np.array_equal(nib.astype(np.float32), q_chunked), "the quants must survive the round trip exactly"

    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < 1e-2, f"writer and reader disagree by {rel:.4g} relative L2"


def test_a_source_that_cannot_be_read_as_q4k_still_packs_with_the_right_min_sign():
    """ggml has no Q4_K encoder, so a q8 / q5 / q6 source under a Q4_K target is
    re-quantized onto q4_1's grid and then packed as Q4_K. The two formats disagree on the
    sign of the min -- q4_1 stores an added `m` (<= 0), Q4_K a subtracted magnitude -- so
    the fallback has to negate it. Without that the weights come out mirrored about each
    block's minimum, and nothing downstream would notice."""
    torch = pytest.importorskip("torch")
    gguf = pytest.importorskip("gguf")
    gt = _converter_packer()

    rows, cols = 32, 256
    w = np.random.default_rng(11).normal(0, 0.05, (rows, cols)).astype(np.float32)
    q8 = gguf.quantize(w, gguf.GGMLQuantizationType.Q8_0).copy()
    t = gt.GGUFTensor("w", (cols, rows), q8, gguf.GGMLQuantizationType.Q8_0)

    d, u, qw = t._requantize_to(gguf.GGMLQuantizationType.Q4_K)
    assert (u.numpy() >= 0).all(), "Q4_K's min is a subtracted magnitude, never negative"

    packed = gt.pack_q4k(d, u, qw, row_block_size=32, col_block_size=256, keep_block_in_2D=False)
    chunks = packed.contiguous().view(torch.uint8).numpy().reshape(-1, Q4K)

    from q4nx import dq_chunks_q4_k
    got = dq_chunks_q4_k(chunks).reshape(rows, cols)
    want = (d.numpy()[:, :, None] * qw.numpy().reshape(rows, cols // 32, 32)
            - u.numpy()[:, :, None]).reshape(rows, cols)
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < 1e-2, f"the fallback's values do not survive the Q4_K pack ({rel:.4g})"
