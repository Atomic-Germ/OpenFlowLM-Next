# Traces: OPEN-PACK-CHUNK-FUSE (canonical spec: specs/open-engine/spec.md)
"""GPT-OSS's 2560-byte chunks fused into the pool's 5120-byte ones.

`q4nx-build/configs/gpt-oss.json` is the only config with `col_block_size` 128, so its
containers hold 32 rows x 128 columns per chunk where every other family holds 32 x 256.
The bytes are ordinary q4_1 -- `d[128]` bf16 at 0, `m[128]` at 256, 2048 nibble bytes at
512 -- so a pool chunk is two adjacent ones fused by eight byte-slice copies, and the file
raster is a supertile rather than the plain one every other converter writes.

The reference here is written independently of `recipes/pack.py`: `_dq_half` decodes a
2560-byte chunk from the format definition, and the fused result is read back with the
SHIPPED `q4nx.dq_chunks_q4_1` -- the one the replica and `gemv_q4.h` agree on. A fuse that
interleaved the wrong slices would have to be wrong in both readers identically to pass.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from q4nx import Q4NX, bf16_to_f32, dq_chunks_q4_1
from recipes import pack
from recipes.catalogue import OpRangeError

CH, CH_HALF = 5120, 2560
RG = 4                      # the supertile height of the attention projections and the experts


def _dq_half(chunk: np.ndarray) -> np.ndarray:
    """One 2560-byte chunk -> [32, 4, 32] f32 (row, 32-column block, lane), from the format
    definition rather than from the packer: 4 blocks of 32 columns, meta index `b*32 + r`,
    nibble raster `(r//16)*2048 + b*512 + i*16 + (r%16)`."""
    meta = bf16_to_f32(np.ascontiguousarray(chunk[:512]).view(np.uint16))
    d, mn = meta[:128], meta[128:]
    q = chunk[512:]
    nib = np.empty(4096, np.float32)
    nib[0::2], nib[1::2] = q & 0xF, q >> 4
    out = np.empty((32, 4, 32), np.float32)
    for r in range(32):
        for b in range(4):
            for i in range(32):
                p = (r // 16) * 2048 + b * 512 + i * 16 + (r % 16)
                out[r, b, i] = nib[p] * d[b * 32 + r] + mn[b * 32 + r]
    return out


def _half_chunks(n: int, seed: int) -> np.ndarray:
    """`n` 2560-byte chunks: bf16 d in [2^-9, 2^-8) and bf16 m with a real sign, so every
    value is finite and no block degenerates."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, CH_HALF), np.uint8)
    for c in range(n):
        d = (0x3B00 | rng.integers(0, 256, 128)).astype("<u2")
        m = (0xBB00 | rng.integers(0, 256, 128)).astype("<u2")
        out[c, 0:256] = d.view(np.uint8)
        out[c, 256:512] = m.view(np.uint8)
        out[c, 512:] = rng.integers(0, 256, CH_HALF - 512, dtype=np.uint8)
    return out


# ----------------------------------------------------------------- the fuse itself

def test_the_fused_chunk_reads_back_as_the_two_halves_side_by_side():
    """Blocks 0..3 of the pool chunk are the low 128 columns and 4..7 the high ones, read
    with the pool's own dequantizer against a half-width reader written from the format."""
    src = _half_chunks(2, 7)
    fused = pack.fuse_chunks(src[0:1], src[1:2])
    assert fused.shape == (1, CH)
    got = dq_chunks_q4_1(fused)[0]                  # [32, 8, 32]
    np.testing.assert_array_equal(got[:, 0:4, :], _dq_half(src[0]))
    np.testing.assert_array_equal(got[:, 4:8, :], _dq_half(src[1]))


def test_the_fuse_is_a_byte_permutation_and_nothing_else():
    """No arithmetic: the pool chunk holds exactly the two sources' bytes, each once."""
    src = _half_chunks(2, 11)
    fused = pack.fuse_chunks(src[0:1], src[1:2])[0]
    both = np.concatenate([src[0], src[1]])
    assert np.array_equal(np.sort(fused), np.sort(both))


@pytest.mark.parametrize("wrong_way", ["concatenate", "planes-swapped", "high-d-and-m-swapped"])
def test_three_near_miss_interleavings_do_not_pass(wrong_way):
    """The discrimination. Each of these produces a 5120-byte chunk out of the same bytes
    and a different reading, so the test above is pinning the layout rather than agreeing
    with whatever `fuse_chunks` happens to do."""
    src = _half_chunks(2, 13)
    a, b = src[0], src[1]
    right = dq_chunks_q4_1(pack.fuse_chunks(src[0:1], src[1:2]))
    wrong = np.empty((1, CH), np.uint8)
    if wrong_way == "concatenate":
        wrong[0] = np.concatenate([a, b])
    elif wrong_way == "planes-swapped":              # each source's two row halves exchanged
        wrong[0, 0:256], wrong[0, 256:512] = a[0:256], b[0:256]
        wrong[0, 512:768], wrong[0, 768:1024] = a[256:512], b[256:512]
        wrong[0, 1024:2048], wrong[0, 2048:3072] = a[1536:2560], b[1536:2560]
        wrong[0, 3072:4096], wrong[0, 4096:5120] = a[512:1536], b[512:1536]
    else:                                            # the high half's d and m exchanged
        wrong[0] = right_bytes = pack.fuse_chunks(src[0:1], src[1:2])[0]
        d_hi, m_hi = right_bytes[256:512].copy(), right_bytes[768:1024].copy()
        wrong[0, 256:512], wrong[0, 768:1024] = m_hi, d_hi
    assert not np.array_equal(right, dq_chunks_q4_1(wrong))


def test_fuse_refuses_a_mismatched_pair():
    src = _half_chunks(3, 17)
    with pytest.raises(ValueError, match="low-half chunks against"):
        pack.fuse_chunks(src[0:2], src[2:3])


# ------------------------------------------------------------- the supertile raster

def test_the_supertile_law_is_a_permutation_and_is_not_the_plain_raster():
    nrb, ncol = 8, 23
    idx = pack.supertile_perm(nrb, ncol, RG)
    assert sorted(idx.reshape(-1).tolist()) == list(range(nrb * ncol))
    plain = np.arange(nrb)[:, None] * ncol + np.arange(ncol)[None, :]
    assert not np.array_equal(idx, plain), "a supertile taller than one row block reorders"
    assert np.array_equal(pack.supertile_perm(nrb, ncol, 1), plain)
    # the law itself, spelled out on one entry: row block 5 is supertile 1 at position 1
    assert idx[5, 3] == (1 * ncol + 3) * RG + 1


def test_the_lm_heads_supertile_is_two_row_blocks_not_four():
    """`rg` is 2 for the head and 4 for the attention projections and the experts, and the
    two rasters really differ -- an op that hardwired 4 would misplace every head chunk."""
    nrb, ncol = 8, 23
    assert not np.array_equal(pack.supertile_perm(nrb, ncol, 2), pack.supertile_perm(nrb, ncol, RG))
    assert sorted(pack.supertile_perm(nrb, ncol, 2).reshape(-1).tolist()) == list(range(nrb * ncol))


def test_a_partial_supertile_is_refused():
    with pytest.raises(OpRangeError, match="supertile"):
        pack.supertile_perm(6, 23, RG)          # 6 row blocks is one and a half supertiles


# --------------------------------------------------------------- std_fuse, end to end

IN_DIM, SRC_DIM = 3072, 2944          # the pool's padded width and the container's own
NCOL128 = SRC_DIM // 128              # 23 real column blocks; the pool wants 24
NRB = 8                               # 8 row blocks = 256 output rows = 4 bands
PER_BAND = IN_DIM // 128              # 24 pool chunks per band
NCH = NRB // 2 * PER_BAND             # 4 bands of 24
NAME = "model.layers.0.self_attn.q_proj.weight"


def _write_container(path, raw: np.ndarray):
    b = raw.reshape(-1).tobytes()
    hdr = {NAME: {"dtype": "I8", "shape": [len(b) // CH_HALF, CH_HALF], "data_offsets": [0, len(b)]}}
    blob = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b)
    return path


def _fuse_op(nch=NCH, in_dim=IN_DIM, src_dim=SRC_DIM):
    return {"op": "std_fuse", "tensor": NAME, "dst": 0, "nch": nch,
            "in_dim": in_dim, "src_dim": src_dim, "rg": RG}


@pytest.fixture()
def packed(tmp_path):
    """The container packed through `std_fuse`, plus the source chunks to check it against.
    Q4NX mmaps and has no close(), so the mapping is released here (Windows tmp_path)."""
    src = _half_chunks(NRB * NCOL128, 29)
    m = Q4NX(_write_container(tmp_path / "gptoss.q4nx", src))
    try:
        dst = np.zeros(NCH * CH, np.uint8)
        pack.apply_op(_fuse_op(), m, 0, dst)
        yield src, dst.reshape(NCH, CH)
    finally:
        m.mm.close()
        m.f.close()


def test_every_pool_chunk_holds_the_k_tile_the_band_law_asks_for(packed):
    """The whole point: pool chunk c of a band covers row half c%2 and k-tile c//2, and its
    two 128-column halves are located in the supertile raster, not the plain one."""
    src, pool = packed
    idx = pack.supertile_perm(NRB, NCOL128, RG)
    rb, kt = pack.band_rowblock_ktile(NCH, IN_DIM)
    real, synthetic = 0, 0
    for c in range(NCH):
        got = dq_chunks_q4_1(pool[c:c + 1])[0]
        for half, blocks in ((2 * kt[c], slice(0, 4)), (2 * kt[c] + 1, slice(4, 8))):
            if half < NCOL128:
                np.testing.assert_array_equal(got[:, blocks, :], _dq_half(src[idx[rb[c], half]]))
                real += 1
            else:
                assert np.all(got[:, blocks, :] == 0.0)
                synthetic += 1
    # 23 real column blocks against the 24 the pool wants: one synthesised half per pool
    # chunk at the last k-tile, and there are two of those (both row halves) per band.
    assert synthetic == 2 * (NRB // 2)
    assert real == 2 * NCH - synthetic


def test_the_plain_raster_would_give_a_different_pool(packed):
    """Without the supertile the same op would still produce a full, plausible pool -- the
    failure this whole law exists to prevent. Assert the two differ on real bytes."""
    src, pool = packed
    plain = np.arange(NRB)[:, None] * NCOL128 + np.arange(NCOL128)[None, :]
    rb, kt = pack.band_rowblock_ktile(NCH, IN_DIM)
    ndiff = 0
    for c in range(NCH):
        if 2 * kt[c] + 1 >= NCOL128:
            continue
        as_plain = pack.fuse_chunks(src[plain[rb[c], 2 * kt[c]]][None, :],
                                    src[plain[rb[c], 2 * kt[c] + 1]][None, :])[0]
        ndiff += not np.array_equal(as_plain, pool[c])
    assert ndiff > NCH // 2, "the two rasters agree too often for this fixture to discriminate"


def test_the_synthesised_column_block_is_identically_zero(packed):
    """K = 2944 is 23 column blocks, an odd count, so the 24th is not in the container. It
    is written as 2560 zero bytes, which makes the fuse and the 2944-to-3072 pad one pass:
    d = m = 0 reads as exactly 0.0, not as a small number."""
    _, pool = packed
    last = [c for c in range(NCH) if (c % PER_BAND) // 2 == PER_BAND // 2 - 1]
    assert len(last) == 2 * (NRB // 2)               # both row halves of all four bands
    for c in last:
        got = dq_chunks_q4_1(pool[c:c + 1])[0]
        assert np.all(got[:, 4:8, :] == 0.0)         # the high half is the synthesised one
        assert not np.all(got[:, 0:4, :] == 0.0)     # and the low half is real
        assert np.all(pool[c, 256:512] == 0) and np.all(pool[c, 768:1024] == 0)
        assert np.all(pool[c, 2048:3072] == 0) and np.all(pool[c, 4096:5120] == 0)


def test_the_pool_is_the_padded_width_not_the_containers(packed):
    _, pool = packed
    assert pool.shape == (NCH, CH)
    assert NCH == NRB // 2 * (IN_DIM // 128)


# ------------------------------------------------------------------------ refusals

def test_a_container_whose_chunks_are_not_2560_is_refused(tmp_path):
    b = np.zeros((4, CH), np.uint8).reshape(-1).tobytes()
    hdr = {NAME: {"dtype": "I8", "shape": [len(b) // CH, CH], "data_offsets": [0, len(b)]}}
    blob = json.dumps(hdr).encode()
    p = tmp_path / "wide.q4nx"
    p.write_bytes(struct.pack("<Q", len(blob)) + blob + b)
    m = Q4NX(p)
    try:
        with pytest.raises(ValueError, match="2560-byte chunks"):
            pack.apply_op(_fuse_op(nch=2, in_dim=512, src_dim=512), m, 0, np.zeros(2 * CH, np.uint8))
    finally:
        m.mm.close()
        m.f.close()


def test_a_pool_narrower_than_the_container_is_refused():
    with pytest.raises(OpRangeError, match="drop columns"):
        pack.fuse_perm(24, 2560, 2944, RG)


def test_a_source_width_that_is_not_whole_chunks_is_refused():
    with pytest.raises(OpRangeError, match="128-column chunks"):
        pack.fuse_perm(24, 3072, 2900, RG)


def test_a_pool_width_that_does_not_tile_the_chunk_is_refused():
    """`std_fuse` shares the band law with `std_perm`, so it inherits OPEN-WIDTH-PAD's
    guard: the container's own 2944 is not a legal POOL width either."""
    with pytest.raises(OpRangeError, match="2944"):
        pack.fuse_perm(23, 2944, 2944, RG)


def test_the_chunk_count_has_to_match_the_widths(packed, tmp_path):
    """A tensor with the wrong number of chunks for the width it claims is the error that
    would otherwise read past the end or silently pack a short matrix."""
    src, _ = packed
    m = Q4NX(_write_container(tmp_path / "short.q4nx", src[:-NCOL128]))
    try:
        with pytest.raises(ValueError, match="needs exactly"):
            pack.apply_op(_fuse_op(), m, 0, np.zeros(NCH * CH, np.uint8))
    finally:
        m.mm.close()
        m.f.close()


# ------------------------------------------------- the two interpreters, byte for byte

# The pool `std_fuse` writes over `_lcg_bytes(0x5EEDFACE, NRB * NCOL128 * 2560)`.
# src/open_qwen36/pools_test.cpp builds the same bytes and asserts the same number, so a
# divergence between the NumPy packer and the C++ one fails one of the two tests.
FUSE_POOL_FNV1A = 0x21F7E3B732137CB2


def _lcg_pool(tmp_path):
    from test_pack_plan import _lcg_bytes

    src = _lcg_bytes(0x5EEDFACE, NRB * NCOL128 * CH_HALF).reshape(NRB * NCOL128, CH_HALF)
    m = Q4NX(_write_container(tmp_path / "lcg.q4nx", src))
    try:
        dst = np.zeros(NCH * CH, np.uint8)
        pack.apply_op(_fuse_op(), m, 0, dst)
        return dst
    finally:
        m.mm.close()
        m.f.close()


def test_the_numpy_and_cpp_interpreters_agree_byte_for_byte(tmp_path):
    from test_pack_plan import _fnv1a

    got = _fnv1a(_lcg_pool(tmp_path).tobytes())
    print(f"\nfuse pool fnv1a = 0x{got:016x}")
    assert got == FUSE_POOL_FNV1A, \
        f"the fused pool changed: 0x{got:016x} (pools_test.cpp asserts 0x{FUSE_POOL_FNV1A:016x})"
