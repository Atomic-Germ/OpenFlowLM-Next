# Traces: OPEN-FAMILY-LFM2, OPEN-SHORT-CONV-KERNEL (canonical spec: specs/open-engine/spec.md)
"""The LFM2 recipe: two layer types out of one dense block plus a conv block.

The numbers asserted here were worked out by hand in
`.claude/plans/lfm2-short-conv.md` from the installed container's shapes, before the
module existed. They are the plan's arithmetic, not the code's own output read back.

The load-bearing one is the pool split. A short-conv layer's four projections are all
hidden-square, so its attention-equivalent block is WIDER than the real attention
layer's (which has two square and two narrow); the FFN therefore cannot sit at the same
offset in both layer types, and each carries its own set.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes import lfm2
from recipes import spec as S
from recipes.catalogue import OpRangeError
from recipes.spec import FULL, SHORT_CONV

CONTAINER = Path(r"C:\Users\josha\.flm\models\LFM2-1.2B-NPU2")
HID, FF, TAPS, LAYERS = 2048, 8192, 3, 16
MIB = 1 << 20


@pytest.fixture(autouse=True)
def validated(monkeypatch):
    """The attention tuple and the short_conv point entered the catalogue with the 2026-09-13
    hardware pass, so the recipe must resolve WITHOUT the override - that is what these
    tests now prove."""
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)


def lfm2_config() -> dict:
    """LFM2-1.2B's real config. The shapes below are the container's, so skip rather than
    assert something else when it is not installed."""
    p = CONTAINER / "config.json"
    if not p.is_file():
        pytest.skip(f"no LFM2 container at {CONTAINER}")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def spec():
    return S.HF_FAMILIES["lfm2"](lfm2_config(), None)


def test_the_container_is_the_hybrid_the_plan_describes(spec):
    assert (spec.family, spec.hidden, spec.intermediate, spec.conv_kernel) == ("lfm2", HID, FF, TAPS)
    assert len(spec.layer_types) == LAYERS
    assert [i for i, t in enumerate(spec.layer_types) if t == FULL] == [2, 5, 8, 10, 12, 14]
    assert spec.layer_types.count(SHORT_CONV) == 10


def test_the_conv_block_is_wider_than_the_attention_block_it_replaces(spec):
    """Four hidden-square projections against q + o square and k + v at a quarter. This is
    why the two layer types cannot share FFN offsets."""
    L = lfm2.layout(spec)
    conv_block = L.POOL_SC_UP                       # everything before the FFN starts
    attn_block = L.dense.POOL_UP
    assert conv_block == 4 * 2_621_440 == 10 * MIB
    assert attn_block == 2 * 2_621_440 + 2 * 655_360
    assert conv_block > attn_block


def test_the_pool_is_sized_for_the_wider_layer_type(spec):
    L = lfm2.layout(spec)
    assert L.POOL_BYTES == 40 * MIB                 # the plan's figure
    assert L.POOL_BYTES >= L.dense.POOL_BYTES


def test_the_conv_state_is_the_window_still_in_reach(spec):
    """taps - 1 rows of hidden f32 -- the two earlier tokens the third tap still sees."""
    L = lfm2.layout(spec)
    assert L.STATE_BYTES == (TAPS - 1) * HID * 4 == 16384


def test_the_fused_input_projection_becomes_three_ordinary_ones(spec):
    """One [3 * hidden, hidden] tensor read three times at source chunk offsets 0, n, 2n --
    qwen35.py's `chunk0` trick. No permute op, and no new gemv_q4 point."""
    L = lfm2.layout(spec)
    plan = lfm2.pack_plan(spec)
    ops = [o for o in plan["layer_types"][SHORT_CONV]["pool"] if "in_proj" in o["tensor"]]
    assert len(ops) == 3
    assert [o["chunk0"] for o in ops] == [0, L.SC_CHUNKS, 2 * L.SC_CHUNKS]
    assert [o["dst"] for o in ops] == [L.POOL_SC_B, L.POOL_SC_C, L.POOL_SC_U]
    assert {o["nch"] for o in ops} == {L.SC_CHUNKS}
    assert L.SC_CHUNKS == 512                       # 2048 x 2048 in 5120-byte chunks
    assert all(o["op"] == "std_perm" for o in ops)


def test_the_conv_taps_go_in_tap_major(spec):
    """The container stores [hidden, taps]; the core wants [taps, hidden], so that one tap's
    32 consecutive channels are 32 contiguous values rather than a stride-3 gather."""
    L = lfm2.layout(spec)
    consts = lfm2.pack_plan(spec)["layer_types"][SHORT_CONV]["consts"]
    conv = [o for o in consts if o["tensor"].endswith("shortconv.conv.weight")]
    assert len(conv) == 1
    assert conv[0]["op"] == "transpose"
    assert (conv[0]["rows"], conv[0]["cols"], conv[0]["elem"]) == (HID, TAPS, 2)
    assert conv[0]["dst"] == L.CD_CONV


def test_the_attention_layer_is_the_dense_plan_unchanged(spec):
    """An LFM2 attention layer has nothing the dense recipe does not already do."""
    plan = lfm2.pack_plan(spec)
    assert sorted(plan["layer_types"]) == [FULL, SHORT_CONV]
    attn = plan["layer_types"][FULL]
    names = [o["tensor"].rsplit(".", 2)[-2] for o in attn["pool"]]
    assert names == ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]


def test_the_embedding_is_named_the_way_lfm2_ships_it(spec):
    """`model.token_embd.weight`, not `embed_tokens` -- the name every other family uses."""
    assert lfm2.pack_plan(spec)["embed"]["tensor"] == "model.token_embd.weight"


def test_a_conv_layer_takes_no_position_table(spec):
    """It has no rotation, so it takes five buffers where an attention layer takes six."""
    pr = lfm2.programs(spec)
    sc = pr["layer_types"][SHORT_CONV]["program"][0]
    assert sc["kernel"] == "cx"
    assert sc["args"] == ["pool", "xres", "consts", "state", "act"]
    assert "ptab" not in sc["args"]
    assert pr["layer_types"][SHORT_CONV]["buffers"]["state"] == {
        "kind": "linear", "bytes": lfm2.layout(spec).STATE_BYTES}


def test_both_layer_types_get_the_same_consts_and_act_sizes(spec):
    """One BO of each per layer, whatever the layer runs, so both entries must name the
    larger size -- the conv layer's, which carries the taps and four extra f32 vectors."""
    L = lfm2.layout(spec)
    pr = lfm2.programs(spec)
    for lt in (FULL, SHORT_CONV):
        assert pr["layer_types"][lt]["buffers"]["consts"] == L.CD_BYTES
        assert pr["layer_types"][lt]["buffers"]["act"] == L.AD_BYTES
    assert L.CD_BYTES > L.dense.CD_BYTES and L.AD_BYTES > L.dense.AD_BYTES


def test_the_design_is_built_alongside_the_dense_one(spec):
    b = lfm2.builds(spec)
    assert sorted(b) == ["dx", "lm_head_q4", "ln", "short_conv"]
    assert b["short_conv"]["design"] == "short_conv/cx.py"
    assert b["short_conv"]["env"] == {"SC_HID": str(HID), "SC_TAPS": str(TAPS)}


def test_a_spec_with_no_conv_layer_is_refused_by_name(spec):
    """That model is the dense recipe's, and routing it here would build a design it has
    no layer for."""
    import dataclasses
    plain = dataclasses.replace(spec, layer_types=tuple([FULL] * LAYERS))
    with pytest.raises(OpRangeError, match="no short_conv"):
        lfm2.layout(plain)


def test_a_different_tap_count_is_refused_by_name(spec):
    """The conv core holds taps - 1 rows of state; a fourth tap is a different design."""
    import dataclasses
    with pytest.raises(OpRangeError, match="conv_kernel"):
        lfm2.layout(dataclasses.replace(spec, conv_kernel=4))


def test_the_band_law_the_gemv_is_handed_recovers_the_projection_width(spec):
    """cx.py passes `per_band(K)` to gemv_q4_gy, and the kernel derives K back from it as
    256 * per_band / rs (gemv_q4_pool_group_rt). The first LFM2 build divided it by the
    chunks-per-element count as well, so the kernel read a 1024-wide table for a 2048-wide
    projection and B, C and u came out around 1e37 (2026-09-13). The law has to round-trip."""
    from recipes.qwen36moe import CHUNK, PER_CALL, band_bytes, per_band
    for K in (spec.hidden, spec.intermediate):
        assert 256 * per_band(K) // 2 == K
        assert per_band(K) == band_bytes(K) // CHUNK
        assert per_band(K) != band_bytes(K) // CHUNK // PER_CALL
