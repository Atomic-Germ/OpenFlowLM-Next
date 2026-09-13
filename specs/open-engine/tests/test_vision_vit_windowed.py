# Traces: OPEN-VISION-VIT-WINDOWED (canonical spec: specs/open-engine/spec.md)
"""Qwen2.5-VL's vision tower: the windowed one.

Two separate things are checked here, and only the second needs torch.

The windowing math -- which merge units land in which window, in what order, and where
the attention segments start -- is pinned against values worked out by hand from the
geometry (see `.claude/plans/qwen25vl-windowed-vit.md`), not against another component of
this repo. A 6 x 5 merge grid with a 4 x 4 window exercises partial windows on both axes;
a 4 x 4 grid exercises the pad-a-whole-empty-window case the padding arithmetic produces
when the grid divides evenly.

The forward itself is checked against transformers' own
`Qwen2_5_VisionTransformerPretrainedModel` on a small random tower: transformers builds
the weights, the numpy reference reads them, and both run the same synthetic grid. No
container is involved, so this runs offline on any box with torch.
"""
from __future__ import annotations

import numpy as np
import pytest


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        import transformers.models.qwen2_5_vl  # noqa: F401
        return True
    except Exception:
        return False


needs_torch = pytest.mark.skipif(not _have_torch(), reason="needs torch + transformers")

# the small tower every forward check uses: head_dim 16, rotary dim 8 -> 4 (h, w) frequencies
SMALL = dict(depth=4, hidden=64, heads=4, head_dim=16, inter=128, out=32, patch=14, temporal=2,
             merge=2, window=112, fullatt=(2,), eps=1e-6, channels=3)


def test_window_side_is_the_window_in_merge_units():
    import replica_vit_qwen25 as V
    # 112 px / 2 patches per merge unit / 14 px per patch = 4 merge units on a side
    assert V.window_side(112, 2, 14) == 4


def test_window_index_on_a_6x5_merge_grid():
    """12 x 10 patches -> 6 x 5 merge units, 4 x 4 windows: four windows, two of them partial."""
    import replica_vit_qwen25 as V
    idx, cu = V.window_index(12, 10, merge=2, window=112, patch=14)
    assert idx.tolist() == [
        0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18,   # rows 0-3, cols 0-3
        4, 9, 14, 19,                                             # rows 0-3, col 4
        20, 21, 22, 23, 25, 26, 27, 28,                           # rows 4-5, cols 0-3
        24, 29,                                                   # rows 4-5, col 4
    ]
    # segment boundaries are in patches, so merge units x 4: 16, 4, 8, 2 -> 64, 80, 112, 120
    assert cu.tolist() == [0, 64, 80, 112, 120]


def test_window_index_pads_a_whole_empty_window_when_the_grid_divides():
    """8 x 8 patches -> 4 x 4 merge units = exactly one window, but the padding adds three
    empty ones; they must collapse away instead of becoming zero-length segments."""
    import replica_vit_qwen25 as V
    idx, cu = V.window_index(8, 8, merge=2, window=112, patch=14)
    assert idx.tolist() == list(range(16))
    assert cu.tolist() == [0, 64]


def test_window_index_is_a_permutation_of_the_merge_units():
    import replica_vit_qwen25 as V
    for gh, gw in ((12, 10), (8, 8), (2, 2), (16, 24), (6, 18)):
        idx, cu = V.window_index(gh, gw, merge=2, window=112, patch=14)
        n_units = (gh // 2) * (gw // 2)
        assert sorted(idx.tolist()) == list(range(n_units))
        assert cu[0] == 0 and cu[-1] == gh * gw
        assert np.all(np.diff(cu) > 0)


def test_patch_order_is_merge_block_major():
    """The processor's order, unchanged from the Qwen3-VL tower: 2 x 2 blocks, row-major."""
    import replica_vit_qwen25 as V
    p = V.position_ids(4, 4, 2)
    assert p.tolist()[:8] == [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]


@needs_torch
def test_window_index_matches_transformers():
    import torch
    from transformers.vision_utils import get_vision_window_index
    import replica_vit_qwen25 as V
    for gh, gw in ((12, 10), (8, 8), (16, 24), (6, 18)):
        idx, cu = V.window_index(gh, gw, merge=2, window=112, patch=14)
        t_idx, t_cu = get_vision_window_index(torch.tensor([[1, gh, gw]]), spatial_merge_size=2,
                                              window_size=112, patch_size=14)
        assert idx.tolist() == t_idx.tolist()
        assert cu.tolist() == t_cu.tolist()


@needs_torch
def test_numpy_tower_matches_transformers():
    """transformers makes the weights, the numpy forward reads them; 12 x 10 patches."""
    import replica_vit_qwen25 as V
    gh, gw = 12, 10
    m, w = V.random_tower(SMALL, seed=7)
    pixels = np.random.default_rng(11).standard_normal(
        (gh * gw, SMALL["channels"] * SMALL["temporal"] * SMALL["patch"] ** 2), dtype=np.float32)
    y_np = V.vit_forward_np(w, SMALL, pixels, gh, gw)
    y_hf = V.hf_forward(m, pixels, gh, gw)
    assert y_np.shape == (gh * gw // 4, SMALL["out"])
    corr = np.corrcoef(y_np.ravel().astype(np.float64), y_hf.ravel().astype(np.float64))[0, 1]
    assert corr > 0.99999
    assert np.abs(y_np - y_hf).max() < 1e-4 * np.abs(y_hf).max()


@needs_torch
def test_every_layer_full_attention_also_matches_transformers():
    """With no windowed layer the tower is the plain one; the window permutation still
    reorders the rows, so this catches a permutation that is only right by cancellation."""
    import replica_vit_qwen25 as V
    cfg = dict(SMALL, fullatt=(0, 1, 2, 3))
    gh, gw = 12, 10
    m, w = V.random_tower(cfg, seed=5)
    pixels = np.random.default_rng(13).standard_normal(
        (gh * gw, cfg["channels"] * cfg["temporal"] * cfg["patch"] ** 2), dtype=np.float32)
    y_np = V.vit_forward_np(w, cfg, pixels, gh, gw)
    y_hf = V.hf_forward(m, pixels, gh, gw)
    corr = np.corrcoef(y_np.ravel().astype(np.float64), y_hf.ravel().astype(np.float64))[0, 1]
    assert corr > 0.99999


def test_the_activations_are_silu_and_the_exact_gelu():
    """The block MLP is SwiGLU and the merger's is `nn.GELU()`, the erf one -- x * Phi(x),
    checked against the standard normal CDF at points anyone can look up. The tanh
    approximation agrees to about 1e-4, which the oracle comparison would let through, so
    it is pinned here instead."""
    import replica_vit_qwen25 as V
    x = np.array([-1.0, 0.5, 1.0, 2.0], np.float64)
    phi = np.array([0.15865525393145707, 0.6914624612740131, 0.8413447460685429, 0.9772498680518208])
    assert np.allclose(V.gelu_erf(x), x * phi, atol=1e-12)
    assert np.allclose(V.silu(x), x / (1.0 + np.exp(-x)), atol=1e-12)
    tanh_gelu = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    assert np.abs(V.gelu_erf(x) - tanh_gelu).max() > 1e-4


def test_a_segment_only_mixes_its_own_rows():
    """With q = k = 0 every score ties, so each row gets its segment's mean of v. Three
    segments of 4, 2 and 4 rows over v = 0..9 give 1.5, 4.5 and 7.5."""
    import replica_vit_qwen25 as V
    n = 10
    q = np.zeros((n, 1, 1), np.float32)
    v = np.arange(n, dtype=np.float32).reshape(n, 1, 1)
    out = V.segment_attention(q, q, v, np.array([0, 4, 6, 10]), 1.0)
    assert out.ravel().tolist() == [1.5] * 4 + [4.5] * 2 + [7.5] * 4


@needs_torch
def test_ignoring_the_windows_does_not_match_transformers():
    """The check the oracle comparison is worth nothing without. If every block attended
    over the whole image the answer must be visibly wrong, otherwise a passing corr says
    only that the residual stream dominates."""
    import replica_vit_qwen25 as V
    gh, gw = 12, 10
    m, w = V.random_tower(SMALL, seed=7)
    pixels = np.random.default_rng(11).standard_normal(
        (gh * gw, SMALL["channels"] * SMALL["temporal"] * SMALL["patch"] ** 2), dtype=np.float32)
    y_hf = V.hf_forward(m, pixels, gh, gw)
    y_full = V.vit_forward_np(w, dict(SMALL, fullatt=(0, 1, 2, 3)), pixels, gh, gw)
    corr = np.corrcoef(y_full.ravel().astype(np.float64), y_hf.ravel().astype(np.float64))[0, 1]
    assert corr < 0.9
    assert np.abs(y_full - y_hf).max() > 0.1 * np.abs(y_hf).max()


@needs_torch
def test_one_window_grid_is_full_attention():
    """8 x 8 patches fit in a single 4 x 4-unit window, so windowed and full must agree."""
    import replica_vit_qwen25 as V
    gh, gw = 8, 8
    _, w = V.random_tower(SMALL, seed=21)
    pixels = np.random.default_rng(23).standard_normal(
        (gh * gw, SMALL["channels"] * SMALL["temporal"] * SMALL["patch"] ** 2), dtype=np.float32)
    y_win = V.vit_forward_np(w, SMALL, pixels, gh, gw)
    y_full = V.vit_forward_np(w, dict(SMALL, fullatt=(0, 1, 2, 3)), pixels, gh, gw)
    assert np.abs(y_win - y_full).max() < 1e-4 * max(np.abs(y_full).max(), 1e-6)
