# Traces: OPEN-QUANT-Q4K, OPEN-SPEC-DERIVE, OPEN-PACK-PLAN (canonical spec: specs/open-engine/spec.md)
"""A container's own chunk size and tokenizer shape must not stop a spec being derived.

Two readers said "I do not know this" about a file their own packer writes:

  - `recipes/pack.py` transcodes Q4_K (4736-byte) chunks into pool q4_1 chunks
    (`q4k_to_q4_1`), and `model/q4nx.py` reads them -- but spec.py's chunk table
    had no entry for 4736, so `quant_map_from_chunk_sizes` REFUSED the tensor.
    `oflm pack --quant Q4_K` was the documented 35B-MoE recipe, so those
    containers could never reach `oflm add`'s open-kernel path at all.

  - `tokenizer_vocab` read `model.vocab` as a dict. A GGUF with no source repo
    writes the Unigram shape -- a [[token, score], ...] list -- and `.values()`
    on a list is an AttributeError, which is not one of the two exceptions the
    function catches. It aborted `oflm pack --build-spec` and `oflm add`.
"""
from __future__ import annotations

import json

import pytest

from recipes.load import spec_from_model_dir, tokenizer_vocab
from recipes.spec import CHUNK_FORMAT, ModelSpec, quant_map_from_chunk_sizes


def _q4k_config(tmp_path):
    """A packed Qwen3.5-9B directory whose attention is at Q4_K."""
    cfg = {
        "model_type": "qwen3_5", "hidden_size": 4096, "intermediate_size": 12288,
        "num_hidden_layers": 40, "num_attention_heads": 32, "num_key_value_heads": 4,
        "head_dim": 128, "rope_theta": 1000000.0, "vocab_size": 151936,
        "full_attention_interval": 4, "linear_num_key_heads": 8, "linear_num_value_heads": 32,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}}), encoding="utf-8")
    tensors = {}
    for name in ("model.layers.0.self_attn.q_proj.weight",
                 "model.layers.0.self_attn.k_proj.weight",
                 "model.layers.0.self_attn.v_proj.weight",
                 "model.layers.0.self_attn.o_proj.weight",
                 "model.layers.0.linear_attn.qkv_proj.weight",
                 "model.layers.0.mlp.up_proj.weight"):
        tensors[name] = {"dtype": "I8", "shape": [1, 4736], "data_offsets": [0, 4736]}
    tensors["lm_head.weight"] = {"dtype": "I8", "shape": [1, 8704], "data_offsets": [0, 8704]}
    header = json.dumps(tensors).encode()
    (tmp_path / "model.q4nx").write_bytes(len(header).to_bytes(8, "little") + header)
    return tmp_path


def test_a_q4k_container_derives_a_spec(tmp_path):
    """The point of the change: 4736 is not refused, and the role it lands on is
    the one whose pool bytes are identical to a q4_1 container's -- because the
    packer transcodes, not because the deriver was told to guess."""
    d = _q4k_config(tmp_path)
    spec = spec_from_model_dir(d)
    assert spec.family == "qwen35"
    # every role is at pool q4_1, so the canonical map is the bare string and no
    # shipped spec moves: a Q4_K container shares the q4_1 kernel set for its shape.
    assert spec.quant == "q4_1"
    assert spec.spec_hash() == ModelSpec.from_hf_config(
        json.loads((d / "config.json").read_text()), real_vocab=2).spec_hash()


def test_the_q4k_chunk_is_known_to_the_deriver():
    """And the byte count alone is enough: Q4_K is not the ambiguous 2560."""
    assert CHUNK_FORMAT[4736] == "q4_1"
    m = quant_map_from_chunk_sizes("qwen35", {"model.layers.0.self_attn.q_proj.weight": 4736})
    assert m == {}                      # nothing above the default, so nothing is reported


def test_an_unknown_chunk_is_still_refused_by_name():
    """Widening the table must not widen it to 'anything'."""
    with pytest.raises(ValueError, match=r"1300-byte quant chunks"):
        quant_map_from_chunk_sizes("qwen35", {"model.layers.0.self_attn.q_proj.weight": 1300})


def test_the_unigram_tokenizer_shape_counts_its_ids(tmp_path):
    """`model_converter`'s GGUF-with-no-source tokenizer, verbatim."""
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps({
        "model": {"type": "Unigram",
                  "vocab": [["<unk>", 0.0], ["a", -1.5], ["b", -2.0]]},
        "added_tokens": [{"id": 7, "content": "<|endoftext|>"}],
    }), encoding="utf-8")
    # the max id is the added token's, not the vocab's
    assert tokenizer_vocab(p) == 8


def test_the_bpe_shape_still_counts(tmp_path):
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 151935}},
        "added_tokens": [{"id": 151936, "content": "<|endoftext|>"}],
    }), encoding="utf-8")
    assert tokenizer_vocab(p) == 151937


def test_a_missing_or_unreadable_tokenizer_is_none(tmp_path):
    assert tokenizer_vocab(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert tokenizer_vocab(bad) is None
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"model": {"type": "BPE"}}), encoding="utf-8")
    assert tokenizer_vocab(empty) is None
