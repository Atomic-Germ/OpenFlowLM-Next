# Traces: OPEN-ATTN-SINK (canonical spec: specs/open-engine/spec.md)
"""The learned per-head attention sink: one extra logit per head that joins the softmax
denominator and has no value vector behind it, so a head's output weights can sum to less
than one. GPT-OSS carries one per head per layer (`self_attn.sinks`).

Two things are pinned here, both offline and both fp64:

  * `replica_dense.sink_softmax` is the reference, and it agrees with the way GPT-OSS's own
    implementation writes it -- append the sink as one more column, softmax the row, drop
    the column again -- and with transformers' `eager_attention_forward` when torch is
    installed.
  * the kernel's form of the same thing. attn.h runs an ONLINE softmax: a running max and a
    running denominator updated one cached row at a time. A sink is a row whose score is the
    sink logit and whose value vector is zero, so the whole change is the state
    `attn_init_impl` starts from -- `m = sink, l = 1` instead of `m = -1e30, l = 0` -- and
    nothing in the per-row loop moves. That equivalence is the design, so it is asserted
    numerically before any kernel exists.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from replica_dense import sink_softmax

ATTN_H = Path(__file__).resolve().parents[3] / "open_kernels" / "designs" / "attn" / "attn.h"

RNG = np.random.default_rng(20260912)


def _hf_form(s, sink):
    """GPT-OSS's own expression, written out: concatenate the sink onto the score row,
    subtract the row max, softmax, then drop the sink column."""
    row = np.concatenate([np.asarray(s, np.float64), [float(sink)]])
    row = row - row.max()
    e = np.exp(row)
    return (e / e.sum())[:-1]


def _online(s, sink, V):
    """attn.h's state machine, in numpy: attn_init_impl seeds (m, l, o), attn_row_impl
    folds one cached row into it, attn_fin_impl divides. Seeding m with the sink and l
    with 1 is the whole of the sink support; the row loop is untouched."""
    V = np.asarray(V, np.float64)
    m, l = float(sink), 1.0            # attn_init_impl, with a sink
    o = np.zeros(V.shape[1])
    for t, st in enumerate(np.asarray(s, np.float64)):
        mn = max(m, st)
        a, b = np.exp(m - mn), np.exp(st - mn)
        l = l * a + b
        o = o * a + b * V[t]
        m = mn
    return o / l                        # attn_fin_impl


def test_a_zero_sink_is_one_extra_row_of_the_softmax():
    """Two positions at score 0 and a sink at 0: three equal logits, so each real position
    gets exactly a third rather than a half, and a third of the mass goes nowhere."""
    w = sink_softmax([0.0, 0.0], 0.0)
    assert w == pytest.approx([1 / 3, 1 / 3], abs=1e-15)
    assert w.sum() == pytest.approx(2 / 3, abs=1e-15)


def test_the_sink_scales_the_denominator_by_its_own_exponential():
    """A sink of ln 2 counts twice, so the denominator is 1 + 1 + 2 and each position gets
    a quarter. This is the number the kernel has to reproduce."""
    w = sink_softmax([0.0, 0.0], np.log(2.0))
    assert w == pytest.approx([0.25, 0.25], abs=1e-15)


def test_a_sink_far_below_the_scores_is_plain_softmax():
    """The fallback the guard has to reproduce: with no sink the attention is what every
    other family computes, so a family without one must come out bit-for-bit unchanged."""
    s = RNG.normal(0, 4, 64)
    plain = np.exp(s - s.max())
    plain /= plain.sum()
    assert sink_softmax(s, -1e30) == pytest.approx(plain, rel=1e-15, abs=0)


def test_the_reference_agrees_with_gpt_oss_own_expression():
    for _ in range(32):
        s = RNG.normal(0, 6, RNG.integers(1, 200))
        sink = float(RNG.normal(0, 6))
        assert sink_softmax(s, sink) == pytest.approx(_hf_form(s, sink), rel=1e-13, abs=1e-300)


def test_the_weights_sum_to_less_than_one_and_the_gap_is_the_sink_mass():
    """What the sink is FOR: the head can decline to attend. The missing mass is exactly the
    sink's own share of the denominator."""
    s = RNG.normal(0, 2, 40)
    for sink in (-5.0, 0.0, 3.0, 9.0):
        w = sink_softmax(s, sink)
        m = max(s.max(), sink)
        z = np.exp(s - m).sum() + np.exp(sink - m)
        assert w.sum() == pytest.approx(1.0 - np.exp(sink - m) / z, rel=1e-14)
        assert 0.0 < w.sum() < 1.0
    # a sink well above every score leaves almost nothing
    assert sink_softmax(s, 40.0).sum() < 1e-10


def test_the_sink_does_not_change_the_relative_weights():
    """It only rescales: no value vector rides on it, so the ratio between two positions is
    what it was without it. A kernel that got this wrong would still normalise to 1."""
    s = RNG.normal(0, 3, 24)
    a, b = sink_softmax(s, -1e30), sink_softmax(s, 2.5)
    assert (b / a) == pytest.approx(np.full(24, (b / a)[0]), rel=1e-13)


def test_the_online_softmax_seeded_with_the_sink_is_the_closed_form():
    """attn.h's per-row update is unchanged; only the state it starts from moves. 512
    positions with a wide score spread, which is where an online softmax would drift if the
    seeding were wrong (the first real row raises the max off the sink and rescales l)."""
    for _ in range(16):
        n = int(RNG.integers(1, 512))
        s, sink = RNG.normal(0, 8, n), float(RNG.normal(0, 8))
        V = RNG.normal(0, 1, (n, 64))
        assert _online(s, sink, V) == pytest.approx(sink_softmax(s, sink) @ V, rel=1e-12, abs=1e-13)


def test_the_online_form_survives_a_sink_that_never_stops_winning():
    """m stays on the sink for the whole context, so `a` is exactly 1 at every row and the
    rescale branch attn_row_impl skips is never taken. The denominator still has to carry
    the 1 the seeding put there."""
    n = 300
    s, sink = RNG.normal(-60, 1, n), 0.0
    V = RNG.normal(0, 1, (n, 32))
    assert _online(s, sink, V) == pytest.approx(sink_softmax(s, sink) @ V, rel=1e-12, abs=1e-14)
    assert sink_softmax(s, sink).sum() < 1e-20


def test_the_sink_is_not_scaled_by_one_over_sqrt_head_dim():
    """Where this lands in the kernel. GPT-OSS multiplies the q.k scores by 1/sqrt(HD) and
    concatenates the sink AFTER that, so the value seeded into the running max is the raw
    sink -- not the sink over sqrt(HD). attn.h folds the scale into q at head dim 64, which
    is GPT-OSS's, so `sv` is already scaled by the time it meets `ml`."""
    hd = 64
    dots, sink = RNG.normal(0, 20, 16), 1.5
    scaled = dots / np.sqrt(hd)
    assert sink_softmax(scaled, sink) == pytest.approx(_hf_form(scaled, sink), rel=1e-13)
    assert sink_softmax(scaled, sink) != pytest.approx(sink_softmax(scaled, sink / np.sqrt(hd)))


def test_a_windowed_layer_sinks_over_the_window_only():
    """GPT-OSS alternates a 128-row sliding layer with a full one and both carry a sink. The
    sink joins the rows the window keeps, so it is seeded once per token, not per window."""
    n, w = 400, 128
    s, sink = RNG.normal(0, 3, n), 0.7
    V = RNG.normal(0, 1, (n, 16))
    assert _online(s[n - w:], sink, V[n - w:]) == pytest.approx(sink_softmax(s[n - w:], sink) @ V[n - w:],
                                                                rel=1e-12, abs=1e-14)


def test_the_guard_is_off_by_default_and_no_family_raises_it():
    """A family without sinks has to compile what it compiled before. attn.h defaults the
    flag to 0 and nothing emits it -- no recipe carries a SINK knob, so no build key moves.
    The objects still have to be diffed before a GPT-OSS build ships; that is step 1 of the
    procedure in the spec and it has NOT been run."""
    h = ATTN_H.read_text(encoding="utf-8")
    assert "#define ATTN_SINK 0" in h
    from recipes import dense as DR
    from recipes import qwen35 as Q35
    from recipes import qwen36moe as MOE
    for mod in (DR, Q35, MOE):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "ATTN_SINK" not in src and "SINK" not in src, mod.__name__
    assert not [f.name for f in fields(DR.DenseGeometry) if "SINK" in f.name.upper()]


def test_the_sink_rides_in_the_meta_element_not_on_a_fifo():
    """The accounting the design rests on. The meta element is E_A = KVH * HD * 2 bytes and
    holds qn and kn at HD bf16 each; NH more bf16 fit after them when NH <= HD * (KVH - 2).
    GPT-OSS 20B: a 1024-byte element holding 128 + 128 + 128 bytes. attn.h asserts the same
    inequality at compile time."""
    for nh, kvh, hd, fits in ((64, 8, 64, True), (16, 2, 128, False), (32, 8, 128, True)):
        e_a = kvh * hd * 2
        assert (4 * hd + 2 * nh <= e_a) is fits
        assert (nh <= hd * (kvh - 2)) is fits
    assert "4 * kHD + 2 * kNH <= kEA" in ATTN_H.read_text(encoding="utf-8")


def test_against_transformers_own_gpt_oss_attention():
    """The oracle, when torch is installed: GPT-OSS's `eager_attention_forward` verbatim.
    Skipped without torch -- the rest of this file needs neither it nor a model."""
    torch = pytest.importorskip("torch")
    from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward

    nh, kvh, hd, n = 8, 2, 64, 37
    sinks = torch.tensor(RNG.normal(0, 2, nh), dtype=torch.float64)

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sinks = torch.nn.Parameter(sinks)
            self.num_key_value_groups = nh // kvh

    q = torch.tensor(RNG.normal(0, 1, (1, nh, 1, hd)), dtype=torch.float64)
    k = torch.tensor(RNG.normal(0, 1, (1, kvh, n, hd)), dtype=torch.float64)
    v = torch.tensor(RNG.normal(0, 1, (1, kvh, n, hd)), dtype=torch.float64)
    out, _ = eager_attention_forward(_M().eval(), q, k, v, None, scaling=hd ** -0.5)
    got = out[0, 0].detach().numpy()                       # [nh, hd]

    qn, kn, vn = q[0].numpy(), k[0].numpy(), v[0].numpy()
    for h in range(nh):
        kvh_i = h // (nh // kvh)
        s = (kn[kvh_i] @ qn[h, 0]) / np.sqrt(hd)
        assert sink_softmax(s, sinks[h].item()) @ vn[kvh_i] == pytest.approx(got[h], rel=1e-12, abs=1e-14)
