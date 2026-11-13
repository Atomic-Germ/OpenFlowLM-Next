# Traces: OPEN-ATTN-ONE-KV-HEAD (canonical spec: specs/open-engine/spec.md)
"""The attention stream's element accounting with ONE kv head (Gemma 3 1B, Gemma 4)."""
from __future__ import annotations

import dataclasses

import pytest

from recipes import dense as DR
from recipes.catalogue import OpRangeError
from recipes.spec import ModelSpec

from test_dense_stream import FAMILIES
from test_gemma3 import HF_GEMMA3_4B

# A buildable one-kv-head shape: Gemma 3 4B's family at 1B's attention (4 heads over 1 kv).
ONE_KV = {**HF_GEMMA3_4B, "hidden_size": 2048, "intermediate_size": 8192, "num_attention_heads": 4,
          "num_key_value_heads": 1, "vocab_size": 262144}


def geo(cfg):
    spec = ModelSpec.from_hf_config(cfg)
    return spec, DR.layout(spec), DR.geometry(spec)


def test_the_element_is_a_whole_cache_row():
    spec, L, G = geo(ONE_KV)
    assert (L.E_A, L.KV_ROW, L.PTAB_ROW) == (1024, 1024, 2048)
    assert (G.PTAB_ELEMS, G.PTAB_CS_ELEM) == (2, 1)
    assert (G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS) == (1, 2, 4, 1)


def test_every_fill_is_whole_elements_and_matches_what_the_core_acquires():
    spec, L, G = geo(ONE_KV)
    for what, nbytes in (("meta", L.E_A), ("position record", L.PTAB_ROW), ("q", G.QW * 4),
                         ("k", G.KVW * 4), ("v", G.KVW * 4), ("kv window row", L.KV_ROW)):
        assert nbytes % L.E_A == 0, what
    assert G.Q_AIN_ELEMS == G.QW * 4 // L.E_A and G.K_AIN_ELEMS == G.KVW * 4 // L.E_A
    assert L.KV_ROW // L.E_A == 1, "a cached row is ONE element, [K_t | V_t]"
    assert 2 * G.HD * 2 <= L.E_A, "qn | kn fit the meta element"


def test_cos_sin_cross_the_record_boundary_only_between_vectors():
    """attn.h reads the cos / sin block 32 floats at a time from whichever element holds them."""
    spec, L, G = geo(ONE_KV)
    start, end = 512, 512 + 4 * spec.rotary_dim
    assert end <= L.PTAB_ROW and start < L.E_A < end, "the block straddles the two elements"
    for off in range(start, end, 128):
        assert off // L.E_A == (off + 127) // L.E_A


def test_an_odd_kv_count_above_one_is_still_refused(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    spec = ModelSpec.from_hf_config({**ONE_KV, "num_attention_heads": 6, "num_key_value_heads": 3, "head_dim": 128,
                                     "query_pre_attn_scalar": 128})
    with pytest.raises(OpRangeError, match="odd kv-head count"):
        DR.recipe(spec)


@pytest.mark.parametrize("rot, ptab_elems", [(256, 2), (64, 1)])
def test_the_block_prefill_route_is_left_out(rot, ptab_elems):
    """At rotary dim 64 the position record is one element, so only the one-kv-head check refuses the route."""
    spec = ModelSpec.from_hf_config(ONE_KV)
    spec = dataclasses.replace(spec, rotary_dim=rot, sliding_window=0, layer_types=("dense",) * spec.num_layers)
    assert DR.geometry(spec).PTAB_ELEMS == ptab_elems
    assert DR.gemm_route(spec) is None


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_no_shipped_family_moves(name):
    """The pre-change formulas, for every family with an even kv count."""
    spec, L, G = geo(FAMILIES[name])
    kvw = spec.num_kv_heads * spec.head_dim
    e_a = kvw * 2
    assert (L.E_A, L.KV_ROW, L.PTAB_ROW) == (e_a, 2 * e_a, max(1024, e_a))
    assert (G.PTAB_ELEMS, G.PTAB_CS_ELEM) == (L.PTAB_ROW // e_a, 512 // e_a)
