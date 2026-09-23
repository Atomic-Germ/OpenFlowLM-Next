"""Qwen3.8-27B text tower: existing qwen35 family, hardware gates remain closed.

Traces: OPEN-FAMILY-QWEN35, OPEN-OP-RANGE.
"""
import json
from pathlib import Path

import pytest

from recipes.spec import FULL, LINEAR, ModelSpec


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
