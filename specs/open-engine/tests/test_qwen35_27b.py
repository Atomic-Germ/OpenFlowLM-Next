"""Qwen3.8-27B text tower: existing qwen35 family, hardware gates remain closed.

Traces: OPEN-FAMILY-QWEN35, OPEN-OP-RANGE.
"""
import json
from pathlib import Path

import pytest
import numpy as np

from recipes.spec import FULL, LINEAR, ModelSpec
from recipes import qwen35 as Q35, qwen36moe as Q36
from recipes import pack
from recipes.catalogue import LIMITS, OpRangeError, require


@pytest.fixture
def spec27():
    path = Path(__file__).parent / "fixtures/config_qwen38_27b.json"
    return ModelSpec.from_hf_config(json.loads(path.read_text()))


def test_official_27b_text_geometry(spec27):
    s = spec27
    assert s.family == "qwen35"
    assert (s.hidden, s.intermediate, s.num_layers) == (5120, 17408, 64)
    assert (s.num_heads, s.num_kv_heads, s.head_dim, s.rotary_dim) == (24, 4, 256, 64)
    assert (s.lin_key_heads, s.lin_value_heads) == (16, 48)
    assert (s.lin_key_dim, s.lin_value_dim) == (128, 128)
    assert (s.lin_qkv_dim, s.lin_value_width, s.vocab) == (10240, 6144, 248320)
    assert s.layer_types[:8] == (LINEAR, LINEAR, LINEAR, FULL) * 2
    assert s.layer_types == (LINEAR, LINEAR, LINEAR, FULL) * 16
    assert s.num_experts == 0
    assert s.attn_gate and s.qk_norm
    assert (s.rope_theta, s.conv_kernel, s.activation) == (1e7, 4, "silu")


def test_nested_and_flat_text_towers_derive_identically(spec27):
    path = Path(__file__).parent / "fixtures/config_qwen38_27b.json"
    flat = ModelSpec.from_hf_config(json.loads(path.read_text())["text_config"])
    a, b = spec27.to_dict(), flat.to_dict()
    a.pop("extra")
    b.pop("extra")
    assert a == b


def test_three_xn_chunks_exceed_legacy_side_schedule(spec27):
    assert Q35.xn_side_elems(spec27) == 3
    assert Q35.ab_tiles_per_half(spec27) == [32, 32, 16]
    # Even a single AB bank would exhaust the legacy shared FIFO schedule.
    assert Q35.glue_side_fills(spec27) == 14 > LIMITS["shim_fills"]


def test_full_ffn_table_exceeds_l1_even_with_one_weight_chunk(spec27):
    table = Q36.tab_bytes(spec27.intermediate)
    assert table == 39168
    total = Q36.core_l1(table, Q36.FFN_MS_FLOATS, Q36.DN_SCRATCH_FLOATS, pc=1)
    assert total == 69888 > Q36.L1_BUDGET == 61440


def test_48_heads_need_two_fixed_width_ab_banks(spec27):
    assert Q36.AB_LANES == 32
    banks = Q36.ab_lanes(spec27) // Q36.AB_LANES
    assert banks == 2
    assert [min(32, spec27.lin_value_heads - b * 32) for b in range(banks)] == [32, 16]
    bank_bytes = spec27.hidden * Q36.AB_LANES * 2
    assert bank_bytes == 327680 == 80 * Q36.ELEM


def test_27b_points_are_not_hardware_validated(spec27, monkeypatch):
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    for op, kwargs in (
        ("ln", dict(width=spec27.hidden)),
        ("gemv_q4", dict(K=spec27.hidden)),
        ("gemv_q4", dict(K=spec27.intermediate)),
        ("lm_head_q8", dict(K=spec27.hidden, vocab=spec27.vocab)),
        ("deltanet", dict(heads=spec27.lin_value_heads)),
    ):
        with pytest.raises(OpRangeError, match="outside the validated"):
            require(op, **kwargs)


class BytesModel:
    def __init__(self, data):
        self.data = data

    def raw(self, name):
        return self.data


@pytest.mark.parametrize("hidden", [7, 64, 2560, 5120])
def test_banked_ab_transpose_preserves_every_element_and_zeroes_tail(hidden):
    src = ((np.arange(48 * hidden, dtype=np.uint32) * 37 + 11) % 65536).astype("<u2").reshape(48, hidden)
    size = 2 * hidden * 32 * 2
    dst = np.full(size + 16, 0xAB, dtype=np.uint8)
    op = dict(op="transpose_banked", tensor="w", rows=48, cols=hidden, elem=2, dst=8)
    pack.apply_op(op, BytesModel(src.tobytes()), 0, dst)
    got = dst[8:-8].view("<u2").reshape(2, hidden, 32)
    np.testing.assert_array_equal(got[0], src[:32].T)
    np.testing.assert_array_equal(got[1, :, :16], src[32:].T)
    assert not got[1, :, 16:].any()
    assert np.all(dst[:8] == 0xAB) and np.all(dst[-8:] == 0xAB)
    for head in (31, 32, 47):
        np.testing.assert_array_equal(got[head // 32, :, head % 32], src[head])


@pytest.mark.parametrize("tail", range(1, 33))
def test_banked_transpose_all_active_tails(tail):
    heads, hidden = 32 + tail, 13
    src = np.arange(heads * hidden, dtype="<u2").reshape(heads, hidden)
    dst = np.full(2 * hidden * 32 * 2, 0xAB, dtype=np.uint8)
    pack.apply_op(dict(op="transpose_banked", tensor="w", rows=heads, cols=hidden,
                       elem=2, dst=0), BytesModel(src.tobytes()), 0, dst)
    got = dst.view("<u2").reshape(2, hidden, 32)
    for head in range(heads):
        np.testing.assert_array_equal(got[head // 32, :, head % 32], src[head])
    assert not got[1, :, tail:].any()


@pytest.mark.parametrize("change,match", [
    ({"rows": 0}, "rows"), ({"cols": 0}, "cols"),
    ({"elem": 0}, "elem"), ({"dst": -1}, "destination"),
    ({"dst": 1}, "destination"), ({"rows": 49}, "tensor"),
])
def test_banked_transpose_rejects_invalid_geometry_before_writing(change, match):
    dst = np.full(2 * 64 * 32 * 2, 0xAB, dtype=np.uint8)
    op = dict(op="transpose_banked", tensor="w", rows=48, cols=64, elem=2, dst=0)
    op.update(change)
    with pytest.raises(ValueError, match=match):
        pack.apply_op(op, BytesModel(bytes(48 * 64 * 2)), 0, dst)
    assert np.all(dst == 0xAB)


def test_wide_heads_select_banked_pack_without_changing_legacy_ops(spec27):
    # Isolate AB packing from the still-unimplemented 17408-wide FFN.
    narrow_ffn = ModelSpec.from_dict(dict(spec27.to_dict(), intermediate=8192))
    ops = Q35.pack_plan(narrow_ffn)["layer_types"][LINEAR]["consts"]
    ab = [o for o in ops if "ssm_alpha_proj" in o.get("tensor", "") or
          "ssm_beta_proj" in o.get("tensor", "")]
    assert len(ab) == 2
    assert all(o["op"] == "transpose_banked" and o["rows"] == 48 and
               o["cols"] == 5120 and "dst_rows" not in o for o in ab)
    assert ab[1]["dst"] - ab[0]["dst"] == 2 * 327680
