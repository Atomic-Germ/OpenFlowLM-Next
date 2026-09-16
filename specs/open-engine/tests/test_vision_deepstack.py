# Traces: OPEN-VISION-VIT-REF, OPEN-VISION-EMBED, OPEN-VISION-VIT-CONFIG
# (canonical spec: specs/open-engine/spec.md)
"""Qwen3-VL's deepstack: three extra vision mergers, and where their output lands.

The tower `replica_vit.py` implements is Qwen3-VL's, and it was validated with deepstack
turned off because the 35B does not use it. Qwen3-VL does. `replica_deepstack.py` adds the
three extra mergers and the decoder-side injection, in fp64.

The oracle is transformers' own `Qwen3VLVisionModel` built with random weights:
transformers makes the tensors, the reference reads them, and both run the same grid.
Nothing in this repo sits on both sides, and no container is needed -- which matters here,
because no Qwen3-VL container has ever been on the machine this was written on.

Three things are pinned separately:

* the forward, against transformers (needs torch);
* the two facts a port gets wrong silently -- which block a feature is tapped after, and
  which decoder layer it is added to -- with a control that fails when they are wrong;
* the geometry a container with no `vision_config` cannot supply, which is the reason
  Qwen3-VL-4B-Instruct-NPU2 is still refused by name.
"""
from __future__ import annotations

import numpy as np
import pytest


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        import transformers.models.qwen3_vl  # noqa: F401
        return True
    except Exception:
        return False


needs_torch = pytest.mark.skipif(not _have_torch(), reason="needs torch + transformers")

# A small tower: head_dim 16 -> rotary dim 8 -> 2 (h, w) frequencies. npos 64 = an 8 x 8
# position table, so an 8 x 12 patch grid interpolates rather than sampling it exactly.
SMALL = dict(depth=6, hidden=64, heads=4, head_dim=16, inter=128, out=32, patch=16,
             temporal=2, merge=2, npos=64, eps=1e-6, channels=3, deepstack=(1, 3, 5))
GRID = (8, 12)


def _pixels(cfg, seed=11):
    gh, gw = GRID
    return np.random.default_rng(seed).standard_normal(
        (gh * gw, cfg["channels"] * cfg["temporal"] * cfg["patch"] ** 2), dtype=np.float32)


@pytest.fixture(scope="module")
def tower():
    import replica_deepstack as D
    m, w = D.hf_tower(SMALL, seed=7)
    px = _pixels(SMALL)
    return D, m, w, px, D.hf_forward(m, px, *GRID)


# ---- the forward, against transformers


@needs_torch
def test_the_merged_output_and_all_three_features_match_transformers(tower):
    D, _m, w, px, want = tower
    got = D.vit_forward_deepstack(w, SMALL, px, *GRID)
    gh, gw = GRID
    n_units = gh * gw // SMALL["merge"] ** 2
    assert got["merged"].shape == (n_units, SMALL["out"])
    assert len(got["deepstack"]) == 3
    for name, g, h in [("merged", got["merged"], want["merged"])] + [
            (f"deepstack[{j}]", got["deepstack"][j], want["deepstack"][j]) for j in range(3)]:
        assert g.shape == h.shape, name
        corr = np.corrcoef(g.ravel(), h.ravel())[0, 1]
        assert corr > 0.99999, f"{name}: corr {corr}"
        assert np.abs(g - h).max() < 1e-3 * np.abs(h).max(), name


@needs_torch
def test_a_deepstack_feature_is_the_width_of_the_decoder_not_of_the_tower(tower):
    """The mergers all end at `out_hidden_size`, which is the text model's hidden size --
    that is what lets the feature be added straight onto the residual stream."""
    _D, _m, _w, _px, want = tower
    for f in want["deepstack"]:
        assert f.shape[1] == SMALL["out"] != SMALL["hidden"]


# ---- the tap: index i means AFTER block i


@needs_torch
def test_taking_each_feature_one_block_early_visibly_disagrees(tower):
    """The control. `deepstack_visual_indexes` names the block a feature is taken *after*,
    and an off-by-one here is the one error that still produces plausible numbers -- the
    norm placement cannot be got wrong silently (the test below), but this can."""
    D, _m, w, px, want = tower
    early = tuple(i - 1 for i in SMALL["deepstack"])
    off = D.vit_forward_deepstack(w, dict(SMALL, deepstack=early), px, *GRID)
    corrs = [np.corrcoef(off["deepstack"][j].ravel(), want["deepstack"][j].ravel())[0, 1]
             for j in range(3)]
    assert max(corrs) < 0.99, corrs


@needs_torch
def test_the_last_block_can_be_tapped_and_still_differs_from_the_merged_output(tower):
    """Tapping the final block feeds the same hidden state to two different mergers. They
    must not come out the same, or the deepstack merger is not being used."""
    D, _m, w, px, want = tower
    one = dict(w, deepstack=w["deepstack"][:1])
    got = D.vit_forward_deepstack(one, dict(SMALL, deepstack=(SMALL["depth"] - 1,)), px, *GRID)
    assert len(got["deepstack"]) == 1
    assert np.corrcoef(got["deepstack"][0].ravel(), want["merged"].ravel())[0, 1] < 0.99


# ---- the two merger flavours


@needs_torch
def test_the_two_mergers_normalise_over_different_widths(tower):
    """`use_postshuffle_norm` is the only structural difference, and it shows in the norm's
    size: the tower's merger normalises one patch across `hidden`, a deepstack merger
    normalises a whole merged row across `hidden * merge^2`. Read off transformers' own
    state_dict, so this pins the upstream module rather than our reading of it."""
    _D, m, _w, _px, _want = tower
    sd = m.state_dict()
    assert tuple(sd["merger.norm.weight"].shape) == (SMALL["hidden"],)
    for j in range(3):
        assert tuple(sd[f"deepstack_merger_list.{j}.norm.weight"].shape) == \
            (SMALL["hidden"] * SMALL["merge"] ** 2,)


@needs_torch
def test_a_deepstack_merger_cannot_be_run_through_the_ordinary_mergers_path(tower):
    """The likely port shortcut -- reuse the merger already written -- does not silently
    give wrong numbers, it fails to broadcast. Worth pinning: it is why the norm placement
    needs no numeric control."""
    D, _m, w, px, _want = tower
    got = D.vit_forward_deepstack(w, SMALL, px, *GRID)
    with pytest.raises(ValueError, match="broadcast"):
        D.merger(got["last_hidden"], w["deepstack"][0], SMALL, postshuffle=False)


def test_the_merger_is_the_exact_gelu_not_the_tanh_one():
    """Both mergers use `nn.GELU()`; the block MLPs use `gelu_pytorch_tanh`. The two agree
    to about 1e-3, which an oracle comparison at corr 0.99999 does not separate."""
    import replica_deepstack as D
    x = np.linspace(-3, 3, 401)
    assert np.abs(D.gelu_erf(x) - D.gelu_tanh(x)).max() > 1e-4
    # x * Phi(x) against the standard normal CDF, computed a different way
    from scipy.stats import norm
    assert np.allclose(D.gelu_erf(x), x * norm.cdf(x), atol=1e-12)


# ---- the decoder side


def test_feature_j_is_added_after_decoder_layer_j():
    """transformers gates the injection as `layer_idx in range(len(embeds))` and runs it
    after `hidden_states = decoder_layer(...)`. So three features cover decoder layers
    0, 1 and 2 -- feature 0 does NOT fold into the input embedding, which would put it
    through layer 0's attention and MLP first."""
    import replica_deepstack as D
    assert D.deepstack_layer_map(3) == [0, 1, 2]
    assert D.deepstack_layer_map(0) == []


def test_the_injection_touches_the_image_rows_and_nothing_else():
    import replica_deepstack as D
    hidden = np.arange(24, dtype=np.float64).reshape(6, 4)
    mask = np.array([False, True, True, False, True, False])
    feat = np.ones((3, 4)) * 10.0
    out = D.inject_deepstack(hidden, mask, feat)
    assert np.array_equal(out[~mask], hidden[~mask])          # text rows unchanged
    assert np.array_equal(out[mask], hidden[mask] + 10.0)
    assert np.array_equal(hidden, np.arange(24).reshape(6, 4))  # and the input is not mutated


def test_the_injection_follows_prompt_order_not_image_order():
    """Two images in one prompt: the feature rows are concatenated in prompt order, so the
    mask's True positions and the feature's rows line up one for one."""
    import replica_deepstack as D
    hidden = np.zeros((7, 2))
    mask = np.array([False, True, True, False, True, True, False])
    feat = np.arange(8, dtype=np.float64).reshape(4, 2)
    out = D.inject_deepstack(hidden, mask, feat)
    assert out[1].tolist() == [0.0, 1.0] and out[5].tolist() == [6.0, 7.0]


def test_a_mask_that_does_not_match_the_feature_count_is_named():
    import replica_deepstack as D
    with pytest.raises(ValueError, match="image token count"):
        D.inject_deepstack(np.zeros((4, 2)), np.array([True, False, False, False]), np.zeros((2, 2)))


def test_a_feature_of_the_wrong_width_is_named():
    import replica_deepstack as D
    with pytest.raises(ValueError, match="out_hidden_size"):
        D.inject_deepstack(np.zeros((2, 5)), np.array([True, True]), np.zeros((2, 3)))


# ---- the geometry a container without a vision_config cannot give


@needs_torch
def test_the_weight_shapes_give_most_of_the_geometry_back(tower):
    """Qwen3-VL-4B-Instruct-NPU2 carries no `vision_config`, so the weight file is the only
    description of its tower on disk. Run the derivation over transformers' own state_dict
    shapes -- an independent layout, not one this repo wrote -- and it recovers depth,
    hidden, MLP width, output width, position count and the merge factor."""
    import replica_deepstack as D
    _D, m, _w, _px, _want = tower
    got = D.geometry_from_tensors({k: tuple(v.shape) for k, v in m.state_dict().items()})
    for k in ("depth", "hidden", "inter", "out", "npos", "merge", "channels"):
        assert got[k] == SMALL[k], k
    assert got["n_deepstack"] == len(SMALL["deepstack"])
    assert got["exact"] is True


@needs_torch
def test_the_head_count_and_the_deepstack_indexes_are_not_in_the_weights(tower):
    """The two numbers that have no source in a container without a `vision_config`, which
    is why Qwen3-VL-4B-Instruct-NPU2 stays refused even once the mergers are implemented.
    A tower with twice the heads has byte-identical tensor shapes, and the merger names
    give how many deepstack taps there are but never which blocks they hang off."""
    import replica_deepstack as D
    _D, m, _w, _px, _want = tower
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert set(D.UNKNOWN) == {"heads", "deepstack"}
    assert "heads" not in D.geometry_from_tensors(shapes)
    assert "deepstack" not in D.geometry_from_tensors(shapes)

    other = D.hf_tower(dict(SMALL, heads=8, head_dim=8), seed=7)[0]
    assert {k: tuple(v.shape) for k, v in other.state_dict().items()} == shapes

    moved = D.hf_tower(dict(SMALL, deepstack=(0, 2, 4)), seed=7)[0]
    assert {k: tuple(v.shape) for k, v in moved.state_dict().items()} == shapes


def test_the_tiled_container_layout_gives_a_rounded_bound_not_an_exact_width():
    """A shipped linear is [out/64][in/256][64][256], padded up to whole tiles, so the
    35B's 4304-wide MLP reads back as 4352. The derivation says `exact: False` rather than
    handing that number over as the real width."""
    import replica_deepstack as D
    shapes = {"model.visual.patch_embed.proj.weight": (1152, 3, 2, 16, 16),
              "model.visual.pos_embed.weight": (2304, 1152),
              "model.visual.merger.linear_fc1.weight": (72, 18, 64, 256),
              "model.visual.merger.linear_fc2.weight": (32, 18, 64, 256)}
    shapes.update({f"model.visual.blocks.{i}.mlp.linear_fc1.weight": (68, 5, 64, 256)
                   for i in range(27)})
    got = D.geometry_from_tensors(shapes)
    assert (got["depth"], got["hidden"], got["npos"], got["merge"]) == (27, 1152, 2304, 2)
    assert got["exact"] is False
    assert got["inter"] == 4352 > 4304          # the real width, rounded up to whole tiles
    assert got["out"] == 2048


def test_the_shipped_container_config_still_has_no_vision_config():
    """Reading the real file, not a fixture written from memory. Until this changes, the
    tower's depth, hidden size, head count and deepstack indexes have no source."""
    import json
    from pathlib import Path
    fix = Path(__file__).resolve().parent / "fixtures" / "config_qwen3vl_4b.json"
    cfg = json.loads(fix.read_text(encoding="utf-8"))
    assert "vision_config" not in cfg
    assert cfg["vision_model_weight"] == "vision_weight.q4nx"


def test_the_upstream_config_names_the_deepstack_layers():
    """Qwen/Qwen3-VL-4B-Instruct does carry them, which is where the numbers in
    `replica_deepstack.QWEN3VL_4B` come from."""
    import json
    from pathlib import Path
    import replica_deepstack as D
    fix = Path(__file__).resolve().parent / "fixtures" / "config_qwen3vl_4b_hf.json"
    v = json.loads(fix.read_text(encoding="utf-8"))["vision_config"]
    assert tuple(v["deepstack_visual_indexes"]) == D.QWEN3VL_4B["deepstack"] == (5, 11, 17)
    assert (v["depth"], v["hidden_size"], v["num_heads"]) == \
        (D.QWEN3VL_4B["depth"], D.QWEN3VL_4B["hidden"], D.QWEN3VL_4B["heads"])
    assert v["out_hidden_size"] == D.QWEN3VL_4B["out"]


def test_a_deepstack_config_is_still_refused_by_the_tower_that_cannot_run_it():
    """`replica_vit.vit_forward_np` has no deepstack path. Until the C++ port grows one,
    `vision_config` must keep refusing the config by name rather than dropping the part."""
    import replica_vit as V
    with pytest.raises(ValueError, match="deepstack"):
        V.vision_config_of({"vision_config": {"depth": 24, "hidden_size": 1024, "num_heads": 16,
                                              "deepstack_visual_indexes": [5, 11, 17]}})
