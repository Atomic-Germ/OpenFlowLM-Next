"""A two-term bf16 split of an fp32 value is not fp32, and the layer norm needs it to be.

`splitN` documents itself as accurate to "~16 mantissa bits". `mac_vv` is built on
it, so a norm written as `mac_vv(0, x, w)` computes `(hi + lo(x)) * w` and drops
the third term. On a real 5120-wide residual row that is **2.18e-05 relative**
against a full-fp32 product, and the per-token layer-norm kernel measures
**8.45e-06** at `xn` while the block route (host fp32 norm) is 4.13e-06 for the
WHOLE layer. A three-term split is **exact** for the same data.

These tests pin the arithmetic that justifies changing the kernel, so the change
is driven by a measured premise rather than a hunch. They also pin the earlier
correction: splitting only the SCALAR into two bf16 pieces is accurate to
1.6e-6 and was NOT the defect -- believing otherwise would have produced a fix
that changed nothing.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def bf16(v):
    a = np.asarray(v, np.float32)
    return ((a.view(np.uint32) >> 16) << 16).view(np.float32)


def split2(v):
    h = bf16(v)
    return h.astype(np.float64), bf16(np.asarray(v, np.float32) - h).astype(np.float64)


def split3(v):
    h = bf16(v)
    r = np.asarray(v, np.float32) - h
    m = bf16(r)
    return h.astype(np.float64), m.astype(np.float64), bf16(r - m).astype(np.float64)


def row(seed=0, n=5120):
    """`x` is the fp32 residual -- the whole point is that it is NOT bf16.
    `w` is a bf16 norm weight, as the pool stores it."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n).astype(np.float32)
    w = bf16(rng.normal(size=n).astype(np.float32))
    inv = np.float32(1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-6))
    return x, w, inv


def rel(got, exact):
    return float(np.max(np.abs(got - exact)) / np.max(np.abs(exact)))


def test_two_term_split_is_much_worse_than_three_term():
    x, w, inv = row()
    exact = x.astype(np.float64) * w.astype(np.float64) * float(inv)
    h, l = split2(x)
    h2, m2, l2 = split3(x)
    two = rel((h + l) * w * float(inv), exact)
    three = rel((h2 + m2 + l2) * w * float(inv), exact)
    assert two > 1e-5, f'two-term should be poor, got {two:.3e}'
    assert three < two / 100, f'three-term must be far better, got {three:.3e}'


def test_three_term_split_reconstructs_the_fp32_input_exactly():
    """hi + mid + lo rebuilds a full fp32 value, which two terms cannot."""
    x, _, _ = row()
    h2, m2, l2 = split3(x)
    h, l = split2(x)
    assert np.array_equal((h2 + m2 + l2).astype(np.float32), x)
    assert not np.array_equal((h + l).astype(np.float32), x)


def test_the_measured_npu_error_sits_inside_the_two_term_bound():
    """The 8.45e-06 measured at xn must be explainable by the two-term split and
    NOT by the three-term one, or this premise is wrong."""
    x, w, inv = row()
    exact = x.astype(np.float64) * w.astype(np.float64) * float(inv)
    h, l = split2(x)
    h2, m2, l2 = split3(x)
    two = rel((h + l) * w * float(inv), exact)
    three = rel((h2 + m2 + l2) * w * float(inv), exact)
    assert three < 8.45e-06 < two


def test_scalar_only_split_is_accurate_enough_and_was_not_the_defect():
    """The correction that matters: splitting the SCALAR into two bf16 pieces
    reconstructs it to ~1.6e-6, well below the 8.45e-06 measured. An earlier
    version of this analysis blamed that and would have changed nothing."""
    inv = np.float32(0.0996)
    h = bf16(inv)
    l = bf16(inv - h)
    err = abs(float(inv) - (float(h) + float(l)))
    assert err < 2e-6, f'scalar split error {err:.3e} is too small to explain the defect'
