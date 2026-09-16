# Traces: OPEN-ATTN-QKV-BIAS, OPEN-FAMILY-QWEN2 (canonical spec: specs/open-engine/spec.md)
"""The element accounting designs/dense/dx.py depends on.

Every buffer the attention core reads or writes is a fifo of fixed-size elements, and
the host fills them by BYTE COUNT. Where a fill does not divide into whole elements, or
delivers more than the core acquires, the stream desynchronises by an element and the
layer silently reads the wrong thing -- and it cascades, because the leftover is still
there when the next layer starts.

Two of these held only by accident until Qwen2.5-3B, whose 2 kv heads at head dim 128
give the narrowest attention element any dense family has had (512 B):

  * the position record is 1024 B, so it is TWO elements, not one (ATTN_PTAB_SPLIT);
  * a core owning 16 heads emits eight og elements of two heads each, not one of
    sixteen -- attn_fin writes kOGH = min(NHL, HPO) heads per call.

Both are arithmetic over the spec, so they are checked here for every dense family
rather than found on hardware.
"""
from __future__ import annotations

import pytest

from recipes import dense as DR
from recipes.spec import ModelSpec

from test_gemma3 import HF_GEMMA3_4B
from test_granite import HF_GRANITE_42_3B
from test_hunyuan import HF_HY_MT2_7B
from test_llama3 import HF_LLAMA31_8B, HF_LLAMA32_1B, HF_LLAMA32_3B, HF_NANBEIGE41_3B
from test_phi3 import HF_PHI4_MINI
from test_qwen2 import HF_QWEN25_3B
from test_qwen3_dense import HF_QWEN3_0_6B, HF_QWEN3_1_7B, HF_QWEN3_4B, HF_QWEN3_8B

FAMILIES = {
    "qwen3-0.6b": HF_QWEN3_0_6B, "qwen3-1.7b": HF_QWEN3_1_7B, "qwen3-4b": HF_QWEN3_4B,
    "qwen3-8b": HF_QWEN3_8B, "llama3.1-8b": HF_LLAMA31_8B, "llama3.2-3b": HF_LLAMA32_3B,
    "llama3.2-1b": HF_LLAMA32_1B, "nanbeige4.1-3b": HF_NANBEIGE41_3B, "gemma3-4b": HF_GEMMA3_4B,
    "granite-4.2-3b": HF_GRANITE_42_3B, "phi4-mini": HF_PHI4_MINI, "hunyuan-7b": HF_HY_MT2_7B,
    "qwen2.5-3b": HF_QWEN25_3B,
}


def geo(cfg):
    spec = ModelSpec.from_hf_config(cfg)
    return spec, DR.layout(spec), DR.geometry(spec)


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_every_ain_fill_is_a_whole_number_of_elements(name):
    """What the host pushes down the attention stream, in fill order."""
    spec, L, G = geo(FAMILIES[name])
    for what, nbytes in (("meta", L.E_A), ("position record", L.PTAB_ROW), ("q", G.QW * 4),
                         ("k", G.KVW * 4), ("v", G.KVW * 4), ("kv window row", L.KV_ROW)):
        assert nbytes % L.E_A == 0, f"{what}: {nbytes} B is not whole {L.E_A} B elements"


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_core_acquires_exactly_what_the_fills_deliver(name):
    spec, L, G = geo(FAMILIES[name])
    assert G.PTAB_ELEMS == L.PTAB_ROW // L.E_A
    assert G.Q_AIN_ELEMS == G.QW * 4 // L.E_A
    assert G.K_AIN_ELEMS == G.KVW * 4 // L.E_A
    assert L.KV_ROW // L.E_A == 2, "one cached row is acquired as (K_t, V_t)"


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_og_elements_tile_the_heads_a_core_owns(name):
    """attn_fin writes kOGH = min(NHL, HPO) heads per call, so the og element is that
    wide and a core emits NHL / kOGH of them -- NOT one element of NHL heads, which is
    the same thing only while a core owns exactly one element's worth."""
    spec, L, G = geo(FAMILIES[name])
    ogh = min(G.NHL, G.HPO)
    assert G.NHL % ogh == 0
    n_og = G.NHL // ogh
    assert G.ACORES * n_og * ogh * G.HD * 2 == G.QW * 2, "the og drains cover the o projection"


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_bias_stream_keeps_step_where_there_is_one(name):
    spec, L, G = geo(FAMILIES[name])
    if not G.QKVB:
        assert (L.CD_QB, L.CD_KB, L.CD_VB) == (-1, -1, -1)
        return
    assert G.QW * 2 // (L.E_A // 2) == G.Q_AIN_ELEMS
    assert G.KVW * 2 // (L.E_A // 2) == G.K_AIN_ELEMS
