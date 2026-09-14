"""`Q4NX.embed()`'s argument validation, which nothing else exercises.

The reader's other paths are covered by test_pack_plan / test_quant_* through the
chunk formats, but `embed()` is called only by `open_kernels/model/make_decode.py`
and never by a test -- so the guard it grew (a token id past the end of
`model.embed_tokens.weight`, or a `hidden` that disagrees with the table's own row
width) could be removed or inverted without anything failing.

Both failures are the shape this project has been bitten by repeatedly: the read
succeeds, the vector comes back correctly shaped and full of plausible numbers, and
nothing downstream can tell it is the wrong memory. So the test asserts the REFUSAL,
not just the happy path, and it pins the boundaries on both sides -- `vocab_size - 1`
must work and `vocab_size` must not.

The container is built here rather than loaded: eight bytes of header length, the
safetensors-shaped JSON header, then the rows. That is the whole of what `embed()`
touches, so no model file is needed.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

import q4nx
from q4nx import Q4NX, bf16_to_f32, f32_to_bf16

VOCAB, HIDDEN = 5, 8


def _write_container(path, vocab=VOCAB, hidden=HIDDEN):
    """A minimal q4nx file holding only the embedding table. Row r is r + col/100,
    so a row read at the wrong offset is obvious rather than merely different."""
    rows = np.array([[r + c / 100.0 for c in range(hidden)] for r in range(vocab)],
                    dtype=np.float32)
    raw = f32_to_bf16(rows).astype("<u2").tobytes()
    header = {
        "__metadata__": {"format": "q4nx-test"},
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [vocab, hidden],
            "data_offsets": [0, len(raw)],
        },
    }
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw)
    # what the reader should give back, after the bf16 round trip
    return bf16_to_f32(f32_to_bf16(rows)).astype(np.float64)


@pytest.fixture()
def table(tmp_path):
    """Q4NX mmaps the file and has no close(); on Windows tmp_path cleanup fails while
    the mapping is live, so close it here rather than leaving it to the collector."""
    expected = _write_container(tmp_path / "embed.q4nx")
    q = Q4NX(tmp_path / "embed.q4nx")
    try:
        yield q, expected
    finally:
        q.mm.close()
        q.f.close()


def test_embed_returns_the_row_that_was_written(table):
    q, expected = table
    for token in range(VOCAB):
        got = q.embed(token, HIDDEN)
        assert got.shape == (HIDDEN,)
        np.testing.assert_array_equal(got, expected[token])


def test_embed_accepts_both_boundaries(table):
    """0 and vocab_size - 1 are valid. A guard written `0 < token < vocab_size` or
    `token <= vocab_size` passes the middle of the range and fails exactly here."""
    q, expected = table
    np.testing.assert_array_equal(q.embed(0, HIDDEN), expected[0])
    np.testing.assert_array_equal(q.embed(VOCAB - 1, HIDDEN), expected[VOCAB - 1])


@pytest.mark.parametrize("token", [-1, VOCAB, VOCAB + 1000])
def test_embed_refuses_a_token_outside_the_table(table, token):
    q, _ = table
    with pytest.raises(IndexError) as e:
        q.embed(token, HIDDEN)
    # the message has to name the range, or the caller cannot tell what it got wrong
    assert str(VOCAB) in str(e.value)


def test_embed_refuses_a_hidden_that_is_not_the_row_width(table):
    """`hidden` is the row stride. A wrong one reads a correctly shaped vector out of
    the wrong place -- in range, for a valid token, with no error. It is the failure
    the bounds check on `token` alone does not catch."""
    q, _ = table
    for wrong in (HIDDEN // 2, HIDDEN * 2, HIDDEN + 1):
        with pytest.raises(ValueError) as e:
            q.embed(0, wrong)
        assert str(HIDDEN) in str(e.value)


def test_embed_default_hidden_is_not_a_silent_guess(table):
    """`embed()`'s default is hidden=2048. On a table of another width that must raise
    rather than read 2048 values from an 8-wide row."""
    q, _ = table
    with pytest.raises(ValueError):
        q.embed(0)


def test_the_guards_run_before_any_read(tmp_path):
    """A container whose header promises more rows than the file holds: a rejected
    token must still be rejected, not turned into a short read or a crash."""
    expected = _write_container(tmp_path / "truncated.q4nx")
    p = tmp_path / "truncated.q4nx"
    p.write_bytes(p.read_bytes()[:-HIDDEN * 2])      # drop the last row's bytes
    q = Q4NX(p)
    try:
        with pytest.raises(IndexError):
            q.embed(VOCAB, HIDDEN)
        with pytest.raises(ValueError):
            q.embed(0, HIDDEN * 2)
    finally:
        q.mm.close()
        q.f.close()


def test_module_exposes_what_this_test_relies_on():
    """f32_to_bf16 / bf16_to_f32 are the fixture's round trip. If either moves, the
    equality assertions above would start comparing against the wrong reference."""
    assert hasattr(q4nx, "f32_to_bf16") and hasattr(q4nx, "bf16_to_f32")
    one = np.float32([1.0])
    np.testing.assert_array_equal(bf16_to_f32(f32_to_bf16(one)), one)
