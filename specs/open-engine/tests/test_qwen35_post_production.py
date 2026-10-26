"""Production post output appends FP32 scratch without moving any other ABI."""
from dataclasses import asdict
import json
from unittest.mock import patch

import pytest
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes.qwen35 import recipe, pack_plan
from recipes import qwen36moe as shared
from recipes.spec import ModelSpec
from decode_bundle import load_bundle
from pathlib import Path

SPECS = Path(__file__).resolve().parents[3] / "open_kernels/recipes/specs"


def wide():
    return load_spec(SPECS / "qwen35-27b.json")


def test_wide_post_uses_fp32_and_preserves_previous_offsets():
    r = recipe(wide())
    assert r.linear.POST_F32 and r.linear.OG_ELEMS == 6
    layout = asdict(r.layout)
    expected = dict(A_XN=0, A_QKV=20480, A_Z=61440, A_VEC=86016, A_O=184320,
                    A_OUT=221184, A_RES=241664, A_XM=262144, A_H=282624,
                    A_OUT2=352256, A_OUT2B=372736, STATE_BYTES=3317760)
    assert {k: layout[k] for k in expected} == expected
    assert layout["A_OG"] == 393216
    assert layout["A_BYTES"] == 417792


def test_standard_manifest_describes_the_appended_output():
    m = manifest(wide())
    lt = m["layer_types"]["linear_attention"]
    assert lt["post_output"] == dict(dtype="float32", offset=393216, bytes=24576)
    assert lt["buffers"]["act"] == 417792
    assert "post_output" not in m["layer_types"]["full_attention"]
    assert len(m["builds"]) == 46


def test_post_upgrade_preserves_packing_and_refuses_the_old_bundle(tmp_path):
    spec=wide();current=manifest(spec);packed=pack_plan(spec)
    with patch.object(shared,"linear_post_fp32",return_value=False):
        old=manifest(spec)
        assert pack_plan(spec)==packed
    assert "post_output" not in old["layer_types"]["linear_attention"]
    assert old["layer_types"]["linear_attention"]["buffers"]["act"]==393216
    (tmp_path/"manifest.json").write_text(json.dumps(old))
    (tmp_path/"toolchain.json").write_text("{}")
    with pytest.raises(ValueError,match="recipe"):
        load_bundle(tmp_path,current)


@pytest.mark.parametrize("role", ["linear", "linear_out", "attn", "ffn"])
def test_mixed_q8_does_not_enable_unvalidated_post_format(role):
    data = wide().to_dict()
    data["quant"] = {role: "q8_0"}
    spec = ModelSpec.from_dict(data)
    assert not shared.linear(spec).POST_F32
