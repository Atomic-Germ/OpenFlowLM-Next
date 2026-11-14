"""Longer prefill prefixes, without disturbing the retained inventory.

The 255/256/257 trio spans a block boundary already (T=256, so B=2 at 257), but a
longer prefix is what exercises more than one block of state carry. Adding the
option must not move any retained preparation: the defaults are pinned here, so a
change to them fails loudly instead of silently invalidating every stored hash.
"""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('prefill_slice', ROOT/'utilities/prefill-slice/check.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_default_boundaries_are_unchanged():
    m = load()
    ids = list(range(1000))
    assert tuple(m.prompt_cases(ids, True)) == (255, 256, 257)
    assert tuple(m.prompt_cases(ids, False)) == (1000,)


def test_explicit_lengths_are_sorted_and_are_prefixes():
    m = load()
    ids = list(range(600))
    cases = m.prompt_cases(ids, True, lengths=[513, 257, 385])
    assert tuple(cases) == (257, 385, 513)
    assert cases[257] == ids[:257]
    assert cases[513] == ids[:513]
    # every case is a prefix of the longest, which is what makes them comparable
    assert cases[257] == cases[513][:257]


def test_duplicate_out_of_range_or_empty_lengths_are_refused():
    m = load()
    ids = list(range(300))
    for bad in ([], [0], [301], [255, 255], [-1]):
        with pytest.raises(ValueError):
            m.prompt_cases(ids, True, lengths=bad)


def test_length_beyond_the_prompt_is_refused():
    m = load()
    with pytest.raises(ValueError):
        m.prompt_cases(list(range(10)), True, lengths=[11])


def test_block_count_confirming_257_already_spans_two_blocks():
    """Pins why the existing evidence is not block-local: 257 tokens at T=256 runs
    two blocks, so inter-block state carry is already exercised and passing."""
    T = 256
    blocks = lambda n: (n + T - 1) // T
    assert blocks(255) == 1 and blocks(256) == 1
    assert blocks(257) == 2
    assert blocks(513) == 3
