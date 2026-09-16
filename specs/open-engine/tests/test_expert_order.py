"""OPEN-PACK-EXPERT-ORDER: which slab of the fused expert tensor is gate, up and down.

The container fuses one layer's three expert projections into a single tensor and names
none of them. Two readings are plausible -- three projections concatenated, or gate and up
alternating every 128 rows with down after them -- and they agree on down while swapping
gate and up. A packer that picks wrong feeds the SwiGLU its two halves the wrong way round
and still produces finite activations, so nothing downstream raises.

The lever is that `q4nx-build` writes each expert bias TWICE: as a named `*_proj_bias`
tensor and into byte 128 of every column-block-0 chunk. That distinguishes gate from up
from down by value, so the reading is measured rather than argued. Every check below is
paired with a near-miss that must fail, because an order this regular is easy to reproduce
by accident.

Runs against a checked-in byte fixture (make_gptoss_fixtures.py), so it needs neither the
14.4 GB container nor the network.
"""
# Traces: OPEN-PACK-EXPERT-ORDER
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "open_kernels"))

from model import mxfp4 as M  # noqa: E402
from recipes import pack  # noqa: E402

FIXTURE = HERE / "fixtures" / "gptoss_expert_slabs.npz"

REAL_ROWS = 2880          # the projections' true width; the container pads to nslab * 128
NSLAB = 23                # 2944 rows / 128
NCOL128 = 23              # 2944 columns / 128
RG = 4


@pytest.fixture(scope="module")
def fx():
    d = np.load(FIXTURE, allow_pickle=True)
    return d["bias_bytes"], d["named"], d["experts"], d["shape"]


def bias_of(bias_bytes, e_i, chunk_index):
    """The 32 biases a column-block-0 chunk carries, read through the real chunk reader.

    The fixture holds bytes 128..191 of each chunk -- all that carries a bias -- so the
    chunk is rebuilt around them; `chunk_bias` reads nothing else. With q = 0 the chunk
    index is `slab * NCOL128 * RG + quarter`, which is what recovers the two."""
    slab, quarter = divmod(int(chunk_index), NCOL128 * RG)
    chunk = np.zeros(M.CHUNK, np.uint8)
    chunk[128:192] = bias_bytes[e_i, slab, quarter]
    return M.chunk_bias(chunk)


def named_padded(named, e_i, role_i):
    u = named[e_i, role_i].astype(np.uint32) << 16
    v = np.zeros(NSLAB * 128, np.float32)
    v[:REAL_ROWS] = u.view(np.float32)
    return v


def slab_bias(bias_bytes, e_i, chunks, role_i, sub_slab):
    """The 128 biases of one projection's own slab, as the chunk padding carries them."""
    return np.concatenate([bias_of(bias_bytes, e_i, chunks[role_i, sub_slab * RG + f, 0])
                           for f in range(RG)])


def test_the_fixture_is_the_container_the_header_records(fx):
    _, _, _, shape = fx
    assert shape.tolist() == [32, 3 * NSLAB, NCOL128, RG, M.CHUNK]


def test_gate_and_up_alternate_and_down_follows():
    """The order itself, as arithmetic: 0,2,4.. is gate, 1,3,5.. is up, 46.. is down."""
    s = pack.expert_slabs(NSLAB)
    assert s.shape == (3, NSLAB)
    assert s[0].tolist() == list(range(0, 2 * NSLAB, 2))
    assert s[1].tolist() == list(range(1, 2 * NSLAB, 2))
    assert s[2].tolist() == list(range(2 * NSLAB, 3 * NSLAB))
    assert sorted(s.reshape(-1).tolist()) == list(range(3 * NSLAB)), "every slab used once"


def test_the_chunk_map_is_a_permutation_of_one_expert():
    """Composed with the supertile raster it must still hit every chunk exactly once -- a
    map that aliases would pack one slab's bytes over another's with nothing raised."""
    c = pack.expert_chunks(NSLAB, NCOL128, RG)
    assert c.shape == (3, NSLAB * RG, NCOL128)
    assert sorted(c.reshape(-1).tolist()) == list(range(3 * NSLAB * RG * NCOL128))


def test_every_slab_carries_the_bias_the_law_predicts(fx):
    """The lever, over all 69 slabs of two experts: the bias in the chunk padding equals
    the named tensor's rows for the role and the slab `expert_chunks` says are there."""
    bias_bytes, named, experts, _ = fx
    chunks = pack.expert_chunks(NSLAB, NCOL128, RG)
    for e_i in range(len(experts)):
        for role_i, role in enumerate(pack.EXPERT_ROLES):
            want = named_padded(named, e_i, role_i)
            for s in range(NSLAB):
                got = slab_bias(bias_bytes, e_i, chunks, role_i, s)
                assert np.array_equal(got, want[s * 128:(s + 1) * 128]), \
                    f"expert {experts[e_i]} {role} slab {s}"


def test_the_rows_past_the_real_width_are_zero_in_the_container(fx):
    """2880 rows padded to 2944, and the converter writes the pad rather than leaving it.
    A packer that assumed otherwise would have to zero the tail itself."""
    bias_bytes, named, experts, _ = fx
    chunks = pack.expert_chunks(NSLAB, NCOL128, RG)
    tail = slab_bias(bias_bytes, 0, chunks, 0, NSLAB - 1)[REAL_ROWS - (NSLAB - 1) * 128:]
    assert tail.size == NSLAB * 128 - REAL_ROWS == 64
    assert np.all(tail == 0)


def test_the_three_roles_are_told_apart_by_value(fx):
    """If the biases were equal, or mostly zero, the test above would pass on any order."""
    _, named, experts, _ = fx
    v = [named_padded(named, 0, i)[:REAL_ROWS] for i in range(3)]
    for i in range(3):
        assert np.abs(v[i]).mean() > 1e-3
        for j in range(i + 1, 3):
            assert not np.array_equal(v[i], v[j])


@pytest.mark.parametrize("name,order", [
    # the other plausible reading: three projections one after another
    ("concatenated", lambda n: np.stack([np.arange(n), n + np.arange(n), 2 * n + np.arange(n)])),
    # the alternation, with gate and up the wrong way round
    ("gate/up swapped", lambda n: np.stack([2 * np.arange(n) + 1, 2 * np.arange(n), 2 * n + np.arange(n)])),
    # down first, then the alternation
    ("down first", lambda n: np.stack([n + 2 * np.arange(n), n + 2 * np.arange(n) + 1, np.arange(n)])),
    # all three alternating, rather than only gate and up
    ("three-way alternation", lambda n: np.stack([3 * np.arange(n), 3 * np.arange(n) + 1,
                                                  3 * np.arange(n) + 2])),
])
def test_a_near_miss_order_does_not_reproduce_the_biases(fx, name, order, monkeypatch):
    bias_bytes, named, experts, _ = fx
    want = order(NSLAB)
    assert not np.array_equal(want, pack.expert_slabs(NSLAB)), f"{name} is the real order"
    monkeypatch.setattr(pack, "expert_slabs", lambda n: want)
    chunks = pack.expert_chunks(NSLAB, NCOL128, RG)
    ok = all(np.array_equal(slab_bias(bias_bytes, 0, chunks, r, s),
                            named_padded(named, 0, r)[s * 128:(s + 1) * 128])
             for r in range(3) for s in range(NSLAB))
    assert not ok, f"{name} must not reproduce the container's biases"


def test_the_plain_raster_does_not_reproduce_the_biases(fx, monkeypatch):
    """The supertile half of the composition, mutated on its own. Every other converter
    writes chunk (rowblock, column) at `rowblock * ncol + column`; this one groups four row
    blocks into a supertile first. Reading GPT-OSS's experts with the ordinary raster is the
    likeliest single mistake here, and it must not land on the right bias."""
    bias_bytes, named, _, _ = fx

    def plain(nrb, ncol128, rg):
        return np.arange(nrb)[:, None] * ncol128 + np.arange(ncol128)[None, :]

    monkeypatch.setattr(pack, "supertile_perm", plain)
    chunks = pack.expert_chunks(NSLAB, NCOL128, RG)
    bad = 0
    for r in range(3):
        for s in range(NSLAB):
            try:
                got = slab_bias(bias_bytes, 0, chunks, r, s)
            except IndexError:
                bad += 1                       # off the end of the fixture: also not a match
                continue
            if not np.array_equal(got, named_padded(named, 0, r)[s * 128:(s + 1) * 128]):
                bad += 1
    assert bad, "the plain raster must not reproduce the container's biases"
