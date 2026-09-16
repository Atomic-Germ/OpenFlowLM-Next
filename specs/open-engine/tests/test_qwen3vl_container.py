"""OPEN-VISION-VIT-FLAT: the layout Qwen3-VL-4B-Instruct-NPU2 actually ships.

The record said this container's vision tensors were declared with a 2-D header shape
"that both untile implementations refuse" and that whether it was the 35B's tile order
with a collapsed header or plain row-major was unsettled -- with a wrong guess giving
image embeddings that look plausible and are wrong.

It is neither. The order is tiles of 64 output rows by 512 input columns, row-major
within a tile and row-major over the tiles, with nothing padded. That was settled by
comparing all 315 tensors element for element against Qwen/Qwen3-VL-4B-Instruct's own
safetensors, not by inspection: 314 match under that rule and the 315th (pos_embed) is
stored at its natural shape.

Most of this runs off the checked-in header alone. The numeric comparison against
transformers needs the 830 MB container and the 3.9 GB upstream shard, so it skips.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "open_kernels" / "model"))

import replica_deepstack as D  # noqa: E402

HEADER = HERE / "fixtures" / "qwen3vl_vision_header.json"
CONTAINER = Path("C:/Users/josha/.flm/models/Qwen3-VL-4B-Instruct-NPU2")

# From Qwen/Qwen3-VL-4B-Instruct's config.json. The container carries none of it.
UPSTREAM_VISION = dict(depth=24, hidden_size=1024, intermediate_size=4096,
                       out_hidden_size=2560, num_position_embeddings=2304,
                       spatial_merge_size=2, patch_size=16, temporal_patch_size=2,
                       in_channels=3, num_heads=16,
                       deepstack_visual_indexes=[5, 11, 17])


@pytest.fixture(scope="module")
def shapes():
    hdr = json.loads(HEADER.read_text(encoding="utf-8"))
    return {k: v["shape"] for k, v in hdr.items()}


# ------------------------------------------------------------------ the layout

def test_the_flat_tile_form_is_recognised_by_its_row_width(shapes):
    tiled = [k for k, s in shapes.items() if D.is_flat_tiled(s)]
    plain = [k for k, s in shapes.items() if not D.is_flat_tiled(s)]
    assert len(tiled) == 104 and len(plain) == 211
    # Every tiled tensor is a whole number of 32768-element rows, by construction.
    assert all(shapes[k][1] == D.FLAT_ROW for k in tiled)
    # pos_embed is 2-D and NOT tiled -- the discriminator is the row width, not the rank.
    assert shapes["model.visual.pos_embed.weight"] == [2304, 1024]
    assert not D.is_flat_tiled(shapes["model.visual.pos_embed.weight"])


def test_untile_flat_is_the_inverse_of_the_tiling_rule():
    out, inn = 128, 1024                       # 2 tiles down, 2 across
    w = np.arange(out * inn, dtype=np.float32).reshape(out, inn)
    nt, kt = out // D.FLAT_TILE_N, inn // D.FLAT_TILE_K
    packed = (w.reshape(nt, D.FLAT_TILE_N, kt, D.FLAT_TILE_K)
                .transpose(0, 2, 1, 3).reshape(-1, D.FLAT_ROW))
    assert np.array_equal(D.untile_flat(packed.reshape(-1), out, inn), w)


def test_untile_flat_refuses_a_shape_that_is_not_whole_tiles():
    with pytest.raises(ValueError, match="whole number"):
        D.untile_flat(np.zeros(64 * 500, np.float32), 64, 500)
    with pytest.raises(ValueError, match="elements"):
        D.untile_flat(np.zeros(10, np.float32), 64, 512)


def test_nothing_in_this_container_is_padded(shapes):
    """The 35B pads every dimension up to a tile; this one does not, which is why the
    element count alone determines the widths."""
    H = UPSTREAM_VISION["hidden_size"]
    cases = {
        "model.visual.blocks.0.attn.qkv.weight": (3 * H, H),
        "model.visual.blocks.0.attn.proj.weight": (H, H),
        "model.visual.blocks.0.mlp.linear_fc1.weight": (UPSTREAM_VISION["intermediate_size"], H),
        "model.visual.blocks.0.mlp.linear_fc2.weight": (H, UPSTREAM_VISION["intermediate_size"]),
    }
    for name, (out, inn) in cases.items():
        assert int(np.prod(shapes[name])) == out * inn, name


# ------------------------------------------------------------ the geometry

def test_geometry_comes_out_of_the_container_and_matches_upstream(shapes):
    cfg = D.geometry_from_flat(shapes)
    assert cfg["exact"] is True
    assert (cfg["depth"], cfg["hidden"], cfg["inter"], cfg["out"]) == (
        UPSTREAM_VISION["depth"], UPSTREAM_VISION["hidden_size"],
        UPSTREAM_VISION["intermediate_size"], UPSTREAM_VISION["out_hidden_size"])
    assert cfg["npos"] == UPSTREAM_VISION["num_position_embeddings"]
    assert cfg["merge"] == UPSTREAM_VISION["spatial_merge_size"]
    assert cfg["channels"] == UPSTREAM_VISION["in_channels"]
    assert cfg["n_deepstack"] == len(UPSTREAM_VISION["deepstack_visual_indexes"])


def test_the_two_numbers_the_container_cannot_give(shapes):
    """A tower with twice the heads has byte-identical shapes, and the merger names say
    how many deepstack taps there are and never which blocks they hang off. Both have to
    come from the model's published config, so neither may be silently defaulted."""
    cfg = D.geometry_from_flat(shapes)
    assert "heads" not in cfg and "head_dim" not in cfg
    assert "deepstack" not in cfg          # n_deepstack is a COUNT, not the indexes


def test_geometry_refuses_a_container_with_no_patch_embed(shapes):
    without = {k: v for k, v in shapes.items() if "patch_embed" not in k}
    with pytest.raises(ValueError, match="patch_embed"):
        D.geometry_from_flat(without)


def test_geometry_refuses_a_patch_embed_that_is_not_three_channels(shapes):
    bad = dict(shapes)
    bad["model.visual.patch_embed.proj.weight"] = [1024, 5, 2, 16, 16]
    with pytest.raises(ValueError, match="channels"):
        D.geometry_from_flat(bad)


# --------------------------------------------- the numbers, against transformers

def _have_files():
    return (CONTAINER / "vision_weight.q4nx").is_file()


def _upstream_shard():
    """Qwen/Qwen3-VL-4B-Instruct's second shard, which holds all 315 visual tensors.

    It is a 3.9 GB download, so it is never in the tree. Point QWEN3VL_UPSTREAM_SHARD at
    it, or leave a copy in the HF cache, and this test runs; otherwise it skips.
    """
    import os

    env = os.environ.get("QWEN3VL_UPSTREAM_SHARD")
    if env and Path(env).is_file():
        return Path(env)
    root = Path.home() / ".cache" / "huggingface" / "hub"
    hits = sorted(root.glob("models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/"
                            "model-00002-of-00002.safetensors"))
    return hits[0] if hits else None


@pytest.mark.skipif(not _have_files(), reason="needs the shipped Qwen3-VL container")
def test_the_container_tower_matches_transformers_with_qwens_own_weights():
    """The whole point: transformers builds the module and loads UPSTREAM's safetensors,
    the replica reads the CONTAINER. Nothing in this repo sits on both sides.

    Recorded result 2026-09-13 on an 8 x 12 grid: merged corr 1.00000000 rel 1.10e-05,
    last_hidden 1.00000000 / 1.38e-05, the three deepstack features 1.00000000 at
    4.46e-07, 8.80e-06 and 7.84e-06. Taking each tap one block early drops deepstack[0]
    to corr 0.6866, so the tap positions are tested rather than assumed.

    The upstream shard is a 3.9 GB download, so this skips unless it is already here.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    shard = _upstream_shard()
    if shard is None:
        pytest.skip("needs Qwen/Qwen3-VL-4B-Instruct's second shard; set "
                    "QWEN3VL_UPSTREAM_SHARD to it")

    from safetensors.torch import load_file
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    hdr = D.read_header(CONTAINER / "vision_weight.q4nx")
    cfg = D.geometry_from_flat({k: v["shape"] for k, v in hdr.items()})
    cfg = dict(cfg, heads=UPSTREAM_VISION["num_heads"],
               head_dim=UPSTREAM_VISION["hidden_size"] // UPSTREAM_VISION["num_heads"],
               deepstack=list(UPSTREAM_VISION["deepstack_visual_indexes"]))
    w = D.load_container(CONTAINER, cfg)

    hf = Qwen3VLVisionConfig(**UPSTREAM_VISION)
    hf._attn_implementation = "eager"
    m = Qwen3VLVisionModel(hf).to(torch.float32).eval()
    sd = {k[len("model.visual."):]: v.to(torch.float32)
          for k, v in load_file(str(shard)).items() if k.startswith("model.visual.")}
    missing, _ = m.load_state_dict(sd, strict=False)
    assert not missing

    gh, gw = 8, 12
    rng = np.random.default_rng(3)
    pix = rng.standard_normal((gh * gw, 3 * 2 * 16 * 16)) * 0.5
    got = D.vit_forward_deepstack(w, cfg, pix, gh, gw)
    with torch.no_grad():
        out = m(torch.tensor(pix, dtype=torch.float32), grid_thw=torch.tensor([[1, gh, gw]]))

    def agree(a, b):
        a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
        assert a.shape == b.shape
        assert np.corrcoef(a, b)[0, 1] > 0.99999
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-3

    agree(got["merged"], out.pooler_output.to(torch.float64).numpy())
    agree(got["last_hidden"], out.last_hidden_state.to(torch.float64).numpy())
    for g, r in zip(got["deepstack"], out.deepstack_features):
        agree(g, r.to(torch.float64).numpy())

    # The control: one block early must be clearly worse, or the tap is not being tested.
    off = D.vit_forward_deepstack(w, dict(cfg, deepstack=[i - 1 for i in cfg["deepstack"]]),
                                  pix, gh, gw)
    c0 = np.corrcoef(np.asarray(off["deepstack"][0]).ravel(),
                     out.deepstack_features[0].to(torch.float64).numpy().ravel())[0, 1]
    assert c0 < 0.9, f"the tap position is not being tested: control corr {c0}"
