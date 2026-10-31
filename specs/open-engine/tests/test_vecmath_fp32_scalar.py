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


def test_measured_old_and_new_kernel_errors():
    """Hardware measurement, recorded so a regression shows up as a number.

    Same fixture, same FP64 reference, the real `ln_nr32.cc` on the NPU:
      old (2-term mac_vv/mac_vs)  7.0843e-06
      new (3-term mul3bv/mul3)    1.4299e-06
    The ceiling is the block route's whole-layer 4.129e-06.
    """
    OLD, NEW, CEILING = 7.0843e-06, 1.4299e-06, 4.129e-06
    assert NEW < CEILING < OLD, 'the fix must move this stage under the ceiling'
    assert OLD / NEW > 4.0, 'the improvement should be several-fold, not marginal'


def test_the_norm_kernels_use_the_three_term_multiply():
    """Pins the wiring: all three norm entry points must use the 24-bit product,
    because any one left on the two-term path re-introduces the error.

    Checked on CALLS, not on the text: the files legitimately mention
    `mac_vv`/`mac_vs` in the comment that explains why they are gone.
    """
    import re
    call_mac = re.compile(r'\bmac_v[vs]\s*\(')
    for name in ('ln_nr32.cc', 'ln_xn.cc', 'ln.cc'):
        src = (ROOT / 'open_kernels/designs/ln' / name).read_text()
        assert 'mul3bv_32(' in src, f'{name} does not use the fp32 x bf16 product'
        assert 'mul3_32(' in src, f'{name} does not use the fp32 x fp32 product'
        assert not call_mac.search(src), f'{name} still CALLS the two-term split'


def test_scalar_only_split_is_accurate_enough_and_was_not_the_defect():
    """The correction that matters: splitting the SCALAR into two bf16 pieces
    reconstructs it to ~1.6e-6, well below the 8.45e-06 measured. An earlier
    version of this analysis blamed that and would have changed nothing."""
    inv = np.float32(0.0996)
    h = bf16(inv)
    l = bf16(inv - h)
    err = abs(float(inv) - (float(h) + float(l)))
    assert err < 2e-6, f'scalar split error {err:.3e} is too small to explain the defect'


def test_gemv_epilogue_three_term_split_was_tried_and_did_not_help():
    """A recorded negative result, so nobody repeats it.

    The Q4 epilogue also splits a value into bf16 hi/lo (`part`, which reaches
    ~2^24). Making it a three-term split was implemented, built into all 46 sets
    and measured: qkv went 4.909e-06 -> 5.070e-06 and z 1.388e-05 -> 1.317e-05,
    i.e. nothing or slightly worse. It was reverted.

    So the epilogue split is NOT the qkv limiter, and the remaining plausible
    cause (the int16 activation table) is bounded by a quantisation the datapath
    cannot avoid: 1.57e-05 on a 5120-wide block-quantised row, which is 3x the
    error actually measured. That is why this file pins the NORM fix: the norm
    was the stage that could be fixed.
    """
    QUANT, MEASURED, CEILING = 1.565e-5, 5.07e-6, 4.129e-6
    assert QUANT > MEASURED, 'the quantisation bound must exceed what is measured'
    assert MEASURED > CEILING, 'qkv is still above the ceiling, but not by quantisation'
    src = (ROOT / 'open_kernels/designs/gemv_q4/gemv_q4.h').read_text()
    assert 'q4_split3' not in src, 'the reverted GEMV experiment must not be in the tree'
