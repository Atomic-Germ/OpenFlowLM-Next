# Traces: OPEN-VISION-VIT-CONFIG (canonical spec: specs/open-engine/spec.md)
"""Where the vision tower's geometry comes from, and which towers the host port refuses.

Three `vision_config` shapes ship in practice, all of them checked here against the real
config.json of a real container:

  QWEN3_6_MOE_*   Qwen3.6-35B-A3B-NPU2
  QWEN3_5_*       Qwen3.5-{0.8,2,4,9}B-NPU2
  plain           the transformers keys (depth / hidden_size / num_heads / ...), which is
                  what raw HF configs carry and what Qwen2.5-VL's container ships

A fourth shape is a container with no `vision_config` at all -- Qwen3-VL-4B-Instruct-NPU2
is like this, because its closed engine hardcodes the tower's numbers in C++ instead of
reading them. That one cannot be served by the host tower and has to say so.

The host tower (replica_vit.py, and the C++ port that mirrors it) implements Qwen3-VL's
full-attention tower without deepstack. A config describing a tower it does not implement
is refused by name rather than run with the extra parts dropped, because dropping them
gives image embeddings that look plausible and are wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIX = Path(__file__).resolve().parent / "fixtures"


def cfg(name: str) -> dict:
    return json.loads((FIX / f"config_{name}.json").read_text(encoding="utf-8"))


def geometry(name: str) -> dict:
    import replica_vit as V
    return V.vision_config_of(cfg(name))


def test_the_35b_prefixed_keys_are_read():
    g = geometry("qwen36_35b")
    assert (g["depth"], g["hidden"], g["heads"], g["head_dim"]) == (27, 1152, 16, 72)
    assert (g["inter"], g["out"], g["patch"], g["temporal"], g["merge"], g["npos"]) == \
        (4304, 2048, 16, 2, 2, 2304)
    assert g["eps"] == 1e-6


def test_the_qwen35_prefixed_keys_are_read():
    g = geometry("qwen35_9b")
    v = cfg("qwen35_9b")["vision_config"]
    assert g["depth"] == v["QWEN3_5_VISION_NUM_LAYERS"]
    assert g["hidden"] == v["QWEN3_5_VISION_EMBED_DIM"]
    assert g["head_dim"] == v["QWEN3_5_VISION_HEAD_DIM"]
    assert g["out"] == v["QWEN3_5_VISION_OUT_HIDDEN_SIZE"]


def test_the_plain_transformers_keys_are_read():
    """head_dim is not one of them: HF vision configs give hidden_size / num_heads."""
    g = geometry("qwen35_0p8b")
    assert (g["depth"], g["hidden"], g["heads"], g["head_dim"]) == (12, 768, 12, 64)
    assert (g["inter"], g["out"], g["npos"], g["channels"]) == (3072, 1024, 2304, 3)
    assert g["eps"] == 1e-6          # the HF block has no epsilon; the tower's default


def test_both_shapes_of_the_same_model_give_the_same_geometry():
    """config_qwen35_0p8b.json is Qwen3.5-0.8B's HF config, _container.json is the
    container OFLM ships for it. The tower is the same tower."""
    assert geometry("qwen35_0p8b") == geometry("qwen35_0p8b_container")


def test_a_container_with_no_vision_config_names_what_was_looked_for():
    """Qwen3-VL-4B-Instruct-NPU2: the closed qwen3vl engine hardcodes the tower, so the
    container carries the weights and none of the numbers."""
    with pytest.raises(ValueError, match="vision_config"):
        geometry("qwen3vl_4b")


def test_a_deepstack_tower_is_refused_by_name():
    """Qwen3-VL's tower feeds vision layers 5/11/17 through extra mergers into the first
    three decoder layers. The host tower does not do that; running it without them is
    wrong, not approximate."""
    with pytest.raises(ValueError, match="deepstack"):
        geometry("qwen3vl_4b_hf")


def test_a_windowed_tower_is_refused_by_name():
    """Qwen2.5-VL's tower is windowed with four full-attention blocks -- a different
    design from the 2-D RoPE full-attention one implemented here."""
    with pytest.raises(ValueError, match="window"):
        geometry("qwen25vl_3b")


def test_an_unimplemented_mlp_activation_is_refused():
    c = cfg("qwen35_0p8b")
    c["vision_config"]["hidden_act"] = "silu"
    import replica_vit as V
    with pytest.raises(ValueError, match="hidden_act"):
        V.vision_config_of(c)
