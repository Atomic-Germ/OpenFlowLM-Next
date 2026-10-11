"""`oflm pack --build-spec` against a packed directory.

This is the seam the whole open-kernel path hinges on: the recipe layer reads the
directory the packer wrote and turns it into a ModelSpec, and `oflm add` then
matches that spec's hash against an installed kernel set. Every one of these used
to raise before the two sides were lined up:

  - a Granite container, because the folded multiplier never reached config.json;
  - a Q4_K container, because 4736 was not in the deriver's chunk table;
  - a GGUF-with-no-source tokenizer.json, because `model.vocab` is a list.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

from q4nx.cli import _build_spec  # noqa: E402


@pytest.fixture(autouse=True)
def recipes_in_a_tmp_dir(tmp_path, monkeypatch):
    """Point OPEN_KERNELS_DIR at a copy, so --build-spec stages there.

    Without this the test writes into the checkout's `recipes/specs/` -- which is
    exactly what the in-tree staging does to a real developer tree, and
    specs/open-engine/tests/test_spec_model_names.py then fails over a spec named
    after a pytest test id.
    """
    import shutil

    root = tmp_path / "open_kernels"
    shutil.copytree(REPO / "open_kernels" / "recipes", root / "recipes")
    monkeypatch.setenv("OPEN_KERNELS_DIR", str(root))
    return root

GRANITE_CFG = {
    "model_type": "granite", "hidden_size": 2560, "intermediate_size": 8192,
    "num_hidden_layers": 40, "num_attention_heads": 40, "num_key_value_heads": 8,
    "head_dim": 64, "rms_norm_eps": 1e-05, "rope_theta": 10000000.0,
    "vocab_size": 100352, "tie_word_embeddings": False, "rope_scaling": None,
    "attention_multiplier": 0.125, "embedding_multiplier": 1.0,
    "residual_multiplier": 1.0, "logits_scaling": 1.0,
}


def _container(path: Path, chunk_bytes: dict) -> None:
    """A model.q4nx: an 8-byte length, the safetensors-shaped JSON header, no data.

    The deriver reads the header only, which is what makes this cheap.
    """
    entries = {}
    for name, ch in chunk_bytes.items():
        entries[name] = {"dtype": "I8", "shape": [1, ch], "data_offsets": [0, 0]}
    blob = json.dumps(entries).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def _dir(tmp_path: Path, config: dict, chunks: dict, vocab=None) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text(json.dumps(
        vocab if vocab is not None else
        {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}},
         "added_tokens": [{"id": 2, "content": "<|endoftext|>"}]}), encoding="utf-8")
    _container(tmp_path / "model.q4nx", chunks)
    return tmp_path


QWEN35_CFG = {
    "model_type": "qwen3_5", "hidden_size": 4096, "intermediate_size": 12288,
    "num_hidden_layers": 40, "num_attention_heads": 32, "num_key_value_heads": 4,
    "head_dim": 128, "rope_theta": 1000000.0, "vocab_size": 151936,
    "full_attention_interval": 4, "linear_num_key_heads": 8,
    "linear_num_value_heads": 32, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def test_a_granite_container_derives_a_spec(tmp_path, capsys):
    """The load-blocker: the folded multiplier is what the recipe checks first."""
    _dir(tmp_path, GRANITE_CFG,
         {"model.layers.0.mlp.gate_proj.weight": 5120,
          "lm_head.weight": 8704})
    _build_spec(str(tmp_path))
    out = capsys.readouterr().out
    assert "family granite" in out, out
    assert "[ERROR]" not in out and "Traceback" not in out

    spec = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert spec["family"] == "granite"
    assert json.loads((tmp_path / "spec.json").read_text())["hidden"] == 2560


def test_a_q4k_container_derives_a_spec(tmp_path, capsys):
    _dir(tmp_path, QWEN35_CFG,
         {"model.layers.0.self_attn.q_proj.weight": 4736,
          "model.layers.0.self_attn.k_proj.weight": 4736,
          "model.layers.0.self_attn.v_proj.weight": 4736,
          "model.layers.0.self_attn.o_proj.weight": 4736,
          "model.layers.0.linear_attn.qkv_proj.weight": 4736,
          "model.layers.0.mlp.up_proj.weight": 4736,
          "lm_head.weight": 8704})
    _build_spec(str(tmp_path))
    out = capsys.readouterr().out
    assert "family qwen35" in out, out
    assert "Traceback" not in out
    # it is q4_1 as far as the pool is concerned, so it shares the shape's set
    spec = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert spec["quant"] == "q4_1"


def test_a_unigram_tokenizer_does_not_crash_the_derivation(tmp_path, capsys):
    """`model_converter._extract_tokenizer_json` writes llama.cpp's Unigram shape:
    `model.vocab` is a [[token, score], ...] list, and `.values()` on it raised
    AttributeError straight out of this function."""
    _dir(tmp_path, QWEN35_CFG,
         {"model.layers.0.self_attn.q_proj.weight": 5120},
         vocab={"model": {"type": "Unigram", "vocab": [["a", 0.0], ["b", -1.0]]},
                "added_tokens": [{"id": 5, "content": "<|endoftext|>"}]})
    _build_spec(str(tmp_path))
    out = capsys.readouterr().out
    assert "Traceback" not in out, out
    # the real vocab came from the max id over the list plus the added token
    spec = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert spec["real_vocab"] == 6


def test_an_unfolded_granite_container_is_refused_by_name(tmp_path, capsys):
    """Sanity: the check the wiring fix satisfies is still the one that runs."""
    cfg = dict(GRANITE_CFG, attention_multiplier=0.015625)
    _dir(tmp_path, cfg, {"model.layers.0.mlp.gate_proj.weight": 5120})
    _build_spec(str(tmp_path))
    out = capsys.readouterr().out
    assert "UNFOLDED" in out, out
    assert "no spec could be derived" in out, out
    assert "Traceback" not in out, out
    assert not (tmp_path / "spec.json").exists()
