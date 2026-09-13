"""What the shipped GPT-OSS-20B container actually holds, and why no recipe can read it.

The plan that preceded this listed four things as "needs the model": where the expert
biases live, whether gate and up arrive split or fused, whether the router weight is
tiled, and whether the container's K is padded. The container is now on disk and answers
all four -- and adds a fifth that outranks them, because the other four stop mattering
until it is solved: the experts are MXFP4, a format nothing in open_kernels/ implements.

The header is checked in, so this needs no 14.4 GB container.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "open_kernels"))

from recipes import pack  # noqa: E402

HEADER = HERE / "fixtures" / "gptoss_container_header.json"


@pytest.fixture(scope="module")
def hdr():
    return json.loads(HEADER.read_text(encoding="utf-8"))


def test_the_experts_are_mxfp4_and_nothing_here_reads_that(hdr):
    """The blocker that outranks every width question.

    One fused U8 tensor per layer holds gate, up and down for all 32 experts, and
    config.json names the method. MXFP4 is 4-bit FLOAT -- sixteen non-uniformly spaced
    levels per element with one shared E8M0 scale per 32 -- so it cannot be transcoded
    into q4_1's sixteen UNIFORM levels the way q8 and Q4_K are. The packer's whole
    chunk vocabulary is those three.
    """
    t = hdr["tensors"]["model.layers.0.ffn_gate_up_down_exps.weight"]
    assert t["dtype"] == "U8"
    assert t["shape"][0] == 32, "all 32 experts in one tensor"
    assert hdr["config"]["quantization_config"]["quant_method"] == "mxfp4"
    # The three chunk sizes the open packer knows. MXFP4 is not among them.
    assert sorted({pack.CH, pack.Q8, pack.Q4K}) == [4736, 5120, 8704]
    assert "ffn_gate_exps.weight" not in str(hdr["tensors"].keys())
    assert "ffn_up_exps.weight" not in str(hdr["tensors"].keys())


def test_the_expert_biases_are_plain_tensors_not_chunk_padding(hdr):
    """The earlier guess was that the converter hides these inside a q4_1 chunk's own
    padding. It does not -- they are named, one per projection, 32 experts wide."""
    for role in ("gate", "up", "down"):
        t = hdr["tensors"][f"model.layers.0.mlp.experts.{role}_proj_bias"]
        assert t["dtype"] == "BF16" and t["shape"] == [32, 2880]


def test_the_sinks_and_the_router_bias_are_named_tensors(hdr):
    assert hdr["tensors"]["model.layers.0.self_attn.sinks.weight"]["shape"] == [64]
    assert hdr["tensors"]["model.layers.0.mlp.router.bias"]["shape"] == [32]


def test_the_router_ships_twice_in_two_different_layouts(hdr):
    """A tiled decode form and a flat prefill form. A recipe has to choose, and the
    earlier design assumed only the tiled one existed."""
    assert hdr["tensors"]["model.layers.0.mlp.router.weight"]["shape"] == [45, 128, 16]
    assert hdr["tensors"]["model.layers.0.mlp.router.weight_prefill"]["shape"] == [32, 2944]


def test_attention_is_i8_not_q4(hdr):
    """Another thing no width fixes: the q/k/v/o projections and the lm_head are I8, so
    even the attention half of this family does not take the q4 GEMV path."""
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert hdr["tensors"][f"model.layers.0.self_attn.{n}.weight"]["dtype"] == "I8"
    assert hdr["tensors"]["lm_head.weight"]["dtype"] == "I8"


def test_the_geometry_is_the_one_the_width_work_assumed(hdr):
    """OPEN-WIDTH-PAD's arithmetic was done against published numbers; the container
    agrees, so that work stands -- it is just not sufficient."""
    c = hdr["config"]
    assert c["hidden_size"] == 2880 and c["intermediate_size"] == 2880
    assert c["num_local_experts"] == 32 and c["num_experts_per_tok"] == 4
    assert c["num_attention_heads"] == 64 and c["num_key_value_heads"] == 8
    assert c["head_dim"] == 64 and c["sliding_window"] == 128
    assert c["swiglu_limit"] == 7.0
    assert c["rope_scaling"]["rope_type"] == "yarn"


def test_the_container_carries_no_quant_format_stamp(hdr):
    """The draft proposal's premise, confirmed on a fourth container."""
    assert hdr["metadata"] is None
