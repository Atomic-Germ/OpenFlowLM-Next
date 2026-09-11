# Traces: OPEN-OP-RANGE (canonical spec: specs/open-engine/spec.md)
"""A recipe asking a template for a point outside its validated set fails at
generation time with the template and the parameter named."""
from __future__ import annotations

import dataclasses

import pytest

from recipes import qwen36moe as Q
from recipes.catalogue import OpRangeError, check_buffer_args, require
from recipes.load import default_spec


def test_the_27b_is_inside_every_validated_set():
    Q.recipe(default_spec())


def test_an_unvalidated_attention_geometry_is_refused_as_a_whole_tuple():
    spec = dataclasses.replace(default_spec(), head_dim=64, rotary_dim=64)
    with pytest.raises(OpRangeError, match=r"attn: \('head_dim', 'num_heads', 'num_kv_heads', "
                                          r"'rotary_dim', 'qk_norm', 'attn_gate', 'qk_norm_post_rope'\) = "
                                          r"\(64, 16, 2, 64, True, True, False\) is outside the "
                                          r"validated combinations"):
        Q.recipe(spec)


def test_a_never_run_combination_of_validated_values_is_refused():
    """The gap the tuple closes: 128 / 16 / 8 are each in a validated tuple, but
    (128, 16, 8) -- a GQA group of 2 at head dim 128 -- entered the catalogue only
    when OPEN-FAMILY-QWEN3's procedure ran on Qwen3-1.7B (2026-09-06); before that
    the per-parameter check had been passing it silently. Its neighbour at 4 kv
    heads is the same kind of never-run combination and is still refused."""
    require("attn", head_dim=128, num_heads=16, num_kv_heads=8, rotary_dim=128,
            rope_theta=1e6, qk_norm=True, attn_gate=False, qk_norm_post_rope=False)
    with pytest.raises(OpRangeError, match=r"\(128, 16, 4, 128, True, False, False\) is outside"):
        require("attn", head_dim=128, num_heads=16, num_kv_heads=4, rotary_dim=128,
                rope_theta=1e6, qk_norm=True, attn_gate=False, qk_norm_post_rope=False)


def test_an_unvalidated_hidden_is_refused_by_the_first_template_that_cannot_take_it():
    """3072 stopped being an example on 2026-09-06, when Llama 3.2 3B put it in the
    `ln` and `gemv_q4` sets; 5120 is the nearest width nobody has built."""
    spec = dataclasses.replace(default_spec(), hidden=5120)
    with pytest.raises(OpRangeError, match=r"ln: width=5120 is outside the validated set "
                                          r"\{1024, 2048, 2560, 3072, 4096\}"):
        Q.recipe(spec)


def test_an_unvalidated_gemv_k_is_refused():
    with pytest.raises(OpRangeError, match=r"gemv_q4: K=5120 is outside the validated set "
                                          r"\{1024, 2048, 2560, 3072, 3584, 4096, 6144, 8192, 9216, 9728, 10240, 10752, 12288, 14336\}"):
        require("gemv_q4", K=5120)


def test_unknown_template_and_parameter():
    with pytest.raises(OpRangeError, match="no template 'conv2d'"):
        require("conv2d", K=1)
    with pytest.raises(OpRangeError, match="no parameter 'colour'"):
        require("ln", colour=1)


def test_more_than_eight_buffer_arguments_is_refused():
    check_buffer_args("ok", ["a"] * 8)
    with pytest.raises(OpRangeError, match="9 buffer arguments"):
        check_buffer_args("too_many", ["a"] * 9)


def test_a_quant_the_gemv_cannot_read_is_refused():
    with pytest.raises(OpRangeError, match="quant='q4_k'"):
        Q.recipe(dataclasses.replace(default_spec(), quant="q4_k"))
