# Traces: OPEN-SHORT-CONV-KERNEL, OPEN-SHORT-CONV-REF (canonical spec: specs/open-engine/spec.md)
"""What `designs/short_conv/sc.h` computes, against the fp64 reference it has to match.

`sc_model` below is a hand transcription of `short_conv_elem`'s loop - the same reads, the
same order, the same window. It is not generated from the header, so it does not follow a
change automatically; what it pins is the CONTRACT between the two, and every clause of
that contract is one an implementation can get wrong while still producing numbers that
look like weights:

  * the window is newest-LAST, so this token's value pairs with the last tap;
  * the weight reaches the core TAP-major, [taps][width], not the [width][taps] the
    container stores;
  * the state that comes out is [old state1, B * u], not [B * u, old state1];
  * the gate C multiplies the convolution's output, and u multiplies B on the way in.

The mutation tests at the bottom are the point of the file: each breaks one clause and
asserts the comparison notices. A test that only checks the correct case would pass just
as happily against a transposed weight.
"""
from __future__ import annotations

import numpy as np
import pytest

from replica_lfm2 import short_conv_step

HID, TAPS, SCW = 256, 3, 32      # small, and SCW is sc.h's vector width so the loop is real


def rng(seed=0):
    return np.random.default_rng(seed)


def inputs(seed=0, hid=HID):
    g = rng(seed)
    W_in = g.standard_normal((3 * hid, hid)) * 0.05
    w_conv = g.standard_normal((hid, TAPS)) * 0.5        # [width, taps], as the container has it
    W_out = g.standard_normal((hid, hid)) * 0.05
    x = g.standard_normal(hid)
    state = g.standard_normal((TAPS - 1, hid)) * 0.3
    return x, W_in, w_conv, W_out, state


def sc_model(B, C, u, w_tap_major, s0, s1, hid=HID, scw=SCW):
    """A transcription of short_conv_elem: the elementwise middle only, over `hid`
    channels in elements of `scw`. The projections either side are the GEMV's job."""
    y = np.zeros(hid)
    n0 = np.zeros(hid)
    n1 = np.zeros(hid)
    for off in range(0, hid, scw):
        sl = slice(off, off + scw)
        bx = B[sl] * u[sl]
        conv = (s0[sl] * w_tap_major[0][sl]
                + s1[sl] * w_tap_major[1][sl]
                + bx * w_tap_major[2][sl])
        y[sl] = C[sl] * conv
        n0[sl] = s1[sl]                                  # the window slides
        n1[sl] = bx
    return y, np.vstack([n0, n1])


def split_projection(x, W_in, hid=HID):
    h = W_in @ x
    return h[:hid], h[hid:2 * hid], h[2 * hid:]


def tap_major(w_conv):
    """What the `transpose` pack op hands the core: [width, taps] -> [taps, width]."""
    return np.ascontiguousarray(w_conv.T)


def run_both(seed=0, **mutate):
    x, W_in, w_conv, W_out, state = inputs(seed)
    want, want_state = short_conv_step(x, W_in, w_conv, W_out, state)
    B, C, u = split_projection(x, W_in)
    w = mutate.get("w", tap_major(w_conv))
    s0, s1 = (state[1], state[0]) if mutate.get("swap_state") else (state[0], state[1])
    if mutate.get("swap_gate"):
        B, C, u = C, B, u
    y, got_state = sc_model(B, C, u, w, s0, s1)
    got = W_out @ y                                      # the out projection, the GEMV's job
    return got, want, got_state, want_state


def test_the_kernel_math_matches_the_fp64_reference():
    got, want, got_state, want_state = run_both()
    assert np.allclose(got, want, rtol=0, atol=1e-9), np.abs(got - want).max()
    assert np.allclose(got_state, want_state, rtol=0, atol=1e-12)


def test_the_state_that_comes_out_is_the_window_minus_its_oldest_row():
    _, _, got_state, want_state = run_both()
    x, W_in, w_conv, W_out, state = inputs(0)
    B, C, u = split_projection(x, W_in)
    assert np.allclose(got_state[0], state[1]), "row 0 is the previous row 1"
    assert np.allclose(got_state[1], B * u), "row 1 is this token's B * u"
    assert np.allclose(got_state, want_state)


@pytest.mark.parametrize("seed", range(4))
def test_it_holds_over_several_draws(seed):
    got, want, _, _ = run_both(seed)
    assert np.allclose(got, want, rtol=0, atol=1e-9)


# ---- the mutations. Each breaks one clause of the contract; each must be caught.

def test_a_weight_left_in_container_order_is_caught():
    """[width, taps] read as [taps, width]. Same bytes, same magnitudes, wrong answer -
    this is the error the whole tap-major transpose exists to prevent."""
    x, W_in, w_conv, W_out, state = inputs(0)
    wrong = np.ascontiguousarray(w_conv.reshape(TAPS, HID))   # a reshape, not a transpose
    got, want, _, _ = run_both(w=wrong)
    assert not np.allclose(got, want, rtol=1e-3, atol=1e-3)


def test_a_window_built_oldest_last_is_caught():
    """Reversing the taps is the same as reversing the window; either way this token stops
    pairing with the last tap."""
    x, W_in, w_conv, W_out, state = inputs(0)
    got, want, _, _ = run_both(w=tap_major(w_conv)[::-1])
    assert not np.allclose(got, want, rtol=1e-3, atol=1e-3)


def test_swapping_the_two_state_rows_is_caught():
    got, want, _, _ = run_both(swap_state=True)
    assert not np.allclose(got, want, rtol=1e-3, atol=1e-3)


def test_gating_with_b_instead_of_c_is_caught():
    """B gates u going in and C gates the convolution coming out; the projection's three
    thirds are interchangeable-looking and this is what tells them apart."""
    got, want, _, _ = run_both(swap_gate=True)
    assert not np.allclose(got, want, rtol=1e-3, atol=1e-3)


def test_the_vector_width_does_not_change_the_answer():
    """sc.h walks the element in 32-lane steps; the result must not depend on that."""
    x, W_in, w_conv, W_out, state = inputs(1)
    B, C, u = split_projection(x, W_in)
    w = tap_major(w_conv)
    a, _ = sc_model(B, C, u, w, state[0], state[1], scw=32)
    b, _ = sc_model(B, C, u, w, state[0], state[1], scw=64)
    assert np.array_equal(a, b)
