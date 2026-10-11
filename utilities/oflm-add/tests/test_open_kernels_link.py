# Traces: OPEN-ADD-KERNEL-LINK (canonical spec: specs/open-engine/spec.md)
#
# oflm-add picks the open kernel set by CAPABILITY, not by model identity: the
# family, the shape keys the set's own `hf_config_check` declares, and the
# per-role weight format. The identity hash still wins when it applies -- it just
# no longer gates. This is the point of q4nx-build + oflm-add together: a user
# converts a finetune and it runs on the kernels its family and shape already
# have, with no toolchain and no rebuild (ROCm/FastFlowLM#690).
import json
import struct
import sys
from pathlib import Path

import pytest

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402

# A Qwen3-dense config small enough to be obviously synthetic; every key the
# recipes' HF deriver reads for model_type "qwen3".
CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 512,
    "vocab_size": 1024,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
}

# The container header the quant-map deriver reads: an 8-byte little-endian
# length then the safetensors JSON. 5120-byte chunks == q4_1 (spec.CHUNK_FORMAT).
Q4NX_HEADER = {
    "__metadata__": {"format": "q4nx"},
    "model.layers.0.self_attn.q_proj.weight": {"dtype": "I8", "shape": [4, 5120]},
    "model.layers.0.mlp.down_proj.weight": {"dtype": "I8", "shape": [4, 5120]},
}

OTHER_HASH = "sha256:" + "de" * 32

# What the recipes' hf_config_check emits for CONFIG -- the set's own declaration
# of what it can serve, which `find_open_kernels` re-runs at link time and
# `open_qwen36::Manifest::check_model` runs fail-closed at load.
HF_CHECK = {
    "head_dim": 64, "hidden_size": 256, "intermediate_size": 512,
    "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 2,
    "vocab_size": 1024,
}


def write_model(root, name="Synth-4B-NPU2", config=None):
    d = root / "models" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config or CONFIG), encoding="utf-8")
    blob = json.dumps(Q4NX_HEADER).encode()
    with open(d / "model.q4nx", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    return d


def write_kernel_set(xclbins, model_name, spec_hash, family="qwen3", quant="q4_1",
                     real_vocab=1024, hf_check=None, declares_check=True):
    """A fake kernel set: a manifest plus the spec.json the exporter writes beside it."""
    d = xclbins / model_name / "open_kernels"
    d.mkdir(parents=True)
    man = {"manifest_version": 1, "family": family, "spec_hash": spec_hash}
    if declares_check:
        man["hf_config_check"] = hf_check if hf_check is not None else HF_CHECK
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (d / "spec.json").write_text(json.dumps(
        {"family": family, "quant": quant, "real_vocab": real_vocab}), encoding="utf-8")
    return d


@pytest.fixture
def model_dir(tmp_path):
    return write_model(tmp_path)


def capability(model_dir):
    cap, note = oflm_add.model_capability(model_dir)
    assert cap, note
    return cap


# ------------------------------------------------------------------ derivation
def test_the_capability_comes_off_the_model_directory(model_dir):
    """Family, shape-relevant keys, weight format and tokenizer count -- the things
    the kernels are actually built for, none of them the weight bytes."""
    cap = capability(model_dir)
    assert cap["family"] == "qwen3"
    assert cap["quant"] == "q4_1"
    assert cap["real_vocab"] == 1024
    assert cap["spec_hash"] != OTHER_HASH
    # Stable: the same directory derives the same capability.
    assert capability(model_dir) == cap


# ------------------------------------------------------- the finetune case
def test_a_finetune_of_a_shipped_shape_runs_on_its_kernels(tmp_path, model_dir):
    """THE regression this fixes. The set was built for a different tokenizer
    (real_vocab 2000, a finetune that added special tokens) -- a different
    spec_hash, the same family and shape and weight format. It links."""
    xclbins = tmp_path / "xclbins"
    shipped = write_kernel_set(xclbins, "Qwen3-4B-NPU2", OTHER_HASH, real_vocab=2000)
    cap = capability(model_dir)
    found, source, _ = oflm_add.find_open_kernels(cap, [xclbins], model_dir.name)
    assert found == shipped
    assert source == "Qwen3-4B-NPU2"


def test_an_exact_hash_still_wins_over_a_near_match(tmp_path, model_dir):
    """The identity hash did not stop matching -- it stopped gating."""
    cap = capability(model_dir)
    xclbins = tmp_path / "xclbins"
    near = write_kernel_set(xclbins, "AAA-Near-NPU2", OTHER_HASH)
    exact = write_kernel_set(xclbins, "ZZZ-Exact-NPU2", cap["spec_hash"])
    found, source, _ = oflm_add.find_open_kernels(cap, [xclbins], model_dir.name)
    assert found == exact and source == "ZZZ-Exact-NPU2"


def test_the_models_own_directory_wins_a_tie(tmp_path, model_dir):
    cap = capability(model_dir)
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "AAA-Other-NPU2", OTHER_HASH)
    mine = write_kernel_set(xclbins, model_dir.name, OTHER_HASH)
    found, source, _ = oflm_add.find_open_kernels(cap, [xclbins], model_dir.name)
    assert found == mine and source == model_dir.name


# ------------------------------------------------------- what still refuses
def test_a_different_family_does_not_link(tmp_path, model_dir):
    xclbins = tmp_path / "xclbins"
    write_kernel_set(xclbins, "Llama-NPU2", OTHER_HASH, family="llama3")
    cap = capability(model_dir)
    assert oflm_add.find_open_kernels(cap, [xclbins], model_dir.name) == (None, None, None)


def test_a_different_shape_does_not_link(tmp_path, model_dir):
    """hf_config_check is the set's own declaration; a shape it does not cover is
    refused exactly as the engine would refuse it at load."""
    xclbins = tmp_path / "xclbins"
    wrong = dict(HF_CHECK, hidden_size=9999)
    write_kernel_set(xclbins, "Wrong-Shape-NPU2", OTHER_HASH, hf_check=wrong)
    cap = capability(model_dir)
    assert oflm_add.find_open_kernels(cap, [xclbins], model_dir.name) == (None, None, None)


def test_a_different_weight_format_ranks_last(tmp_path, model_dir):
    """A q4_1 set under a q8 container still runs, but re-quantises every q8
    projection on the way into the pool -- silently, which is the 35B bug. The
    q8 set must win when both are installed."""
    xclbins = tmp_path / "xclbins"
    same = write_kernel_set(xclbins, "AAA-Q4-NPU2", OTHER_HASH, quant="q4_1")
    other = write_kernel_set(xclbins, "ZZZ-Q8-NPU2", OTHER_HASH, quant={"attn": "q8"})
    cap = capability(model_dir)
    assert cap["quant"] == "q4_1"
    # the set built for THIS container's format wins; the other stays a candidate
    # the engine will run (re-quantising), which _report_match warns about
    found, _, _ = oflm_add.find_open_kernels(cap, [xclbins], model_dir.name)
    assert found == same


def test_a_tokenizer_count_below_the_models_is_not_selected(tmp_path, model_dir):
    """The one identity-ish field that is a hard check: `logits_view()` masks
    every logit above a set's real_vocab to -inf, so a narrower set would make
    this model's high tokens unsampleable. Wider is harmless."""
    xclbins = tmp_path / "xclbins"
    narrow = write_kernel_set(xclbins, "Narrow-NPU2", OTHER_HASH, real_vocab=512)
    cap = capability(model_dir)
    assert cap["real_vocab"] == 1024
    found, _, _ = oflm_add.find_open_kernels(cap, [xclbins], model_dir.name)
    assert found != narrow


def test_no_match_selects_nothing(tmp_path, model_dir):
    xclbins = tmp_path / "xclbins"
    xclbins.mkdir()
    cap = capability(model_dir)
    assert oflm_add.find_open_kernels(cap, [xclbins], model_dir.name) == (None, None, None)


# ------------------------------------------------------- the load-time rule
def test_layer_types_falls_back_to_the_interval():
    """What `check_model` does: a config that omits layer_types but names
    full_attention_interval is checked against the expanded list."""
    cfg = {"hidden_size": 256, "num_hidden_layers": 3, "full_attention_interval": 2,
           "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
           "intermediate_size": 512, "vocab_size": 1024}
    cap = {"config": cfg}
    man = {"hf_config_check": {"layer_types": [
        "linear_attention", "full_attention", "linear_attention"]}}
    ok, _ = oflm_add.passes_hf_config_check(man, cap)
    assert ok
    man = {"hf_config_check": {"layer_types": ["linear_attention"] * 3}}
    ok, why = oflm_add.passes_hf_config_check(man, cap)
    assert not ok and "layer_types" in why


def test_a_missing_key_is_refused_unless_the_manifest_names_a_default():
    cap = {"config": {"hidden_size": 256}}
    man = {"hf_config_check": {"hidden_size": 256, "partial_rotary_factor": 1.0}}
    ok, why = oflm_add.passes_hf_config_check(man, cap)
    assert not ok and "partial_rotary_factor" in why
    man["hf_config_defaults"] = {"partial_rotary_factor": 1.0}
    ok, _ = oflm_add.passes_hf_config_check(man, cap)
    assert ok


def test_model_type_may_name_a_list():
    cap = {"config": {"model_type": "gemma3_text"}}
    ok, _ = oflm_add.passes_hf_config_check(
        {"hf_config_check": {"model_type": ["gemma3", "gemma3_text"]}}, cap)
    assert ok
    ok, why = oflm_add.passes_hf_config_check(
        {"hf_config_check": {"model_type": ["gemma3"]}}, cap)
    assert not ok and "gemma3_text" in why
