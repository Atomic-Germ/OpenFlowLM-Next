"""The package gate must prove coverage per LAYER TYPE, not per layer index.

`pack_plan()` sections are per layer type: `linear_attention` holds the DeltaNet
tensors and `full_attention` the attention ones, and a given layer is only one
of the two. Expanding every `{l}` op over all 64 layers reports 48 phantom
misses for attention on the linear layers (16 full-attention layers x 6 ops).
These tests pin the type-gated expansion and the "nothing silently ignored"
inventory that Phase 16 requires.
"""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('qwen38_package', ROOT/'utilities/qwen38-package/verify.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_ops_keep_their_layer_type():
    m = load()
    plan = {'layer_types': {
        'linear_attention': {'pool': [{'op': 'proj', 'tensor': 'a.{l}.w'}],
                             'consts': [], 'ops': []},
        'full_attention': {'pool': [{'op': 'proj', 'tensor': 'b.{l}.w'}],
                           'consts': [], 'ops': []}},
        'lm_head': {'ops': [{'op': 'lmhead_q8', 'tensor': 'head'}]},
        'embed': {'tensor': 'emb'}, 'norm': {'tensor': 'nrm'}}
    by_tensor = {op['tensor']: (lt, where) for lt, where, op in m.ops_of(plan)}
    assert by_tensor['a.{l}.w'][0] == 'linear_attention'
    assert by_tensor['b.{l}.w'][0] == 'full_attention'
    assert by_tensor['head'][0] is None
    assert set(m.required_tensors(plan)) == {'a.{l}.w', 'b.{l}.w', 'head', 'emb', 'nrm'}


def test_required_tensors_counts_reuse_of_the_fused_q_proj():
    """The fused q_proj is read twice (Q half, then gate half); that is reuse,
    not duplication, so it must be counted and not rejected."""
    m = load()
    plan = {'layer_types': {'full_attention': {'pool': [
        {'op': 'proj', 'tensor': 'q.{l}', 'chunk0': 0},
        {'op': 'proj', 'tensor': 'q.{l}', 'chunk0': 12}], 'consts': [], 'ops': []}},
        'lm_head': {}, 'embed': {}, 'norm': {}}
    info = m.required_tensors(plan)['q.{l}']
    assert info['ops'] == 2
    assert info['layer_types'] == ['full_attention']


def test_expand_substitutes_the_layer_index():
    m = load()
    assert m.expand('model.layers.{l}.mlp.up_proj.weight', 7) == 'model.layers.7.mlp.up_proj.weight'
    assert m.expand('model.norm.weight', 7) == 'model.norm.weight'


def test_type_gated_expansion_skips_irrelevant_layers():
    """Exactly the bug this file exists for: 64 layers, alternating 3 linear +
    1 attention means 48 linear and 16 attention layers, so each op set expands
    to its own count instead of to 64."""
    m = load()
    plan = {'layer_types': {
        'linear_attention': {'pool': [{'op': 'p', 'tensor': 'lin.{l}'}], 'consts': [], 'ops': []},
        'full_attention': {'pool': [{'op': 'p', 'tensor': 'attn.{l}'}], 'consts': [], 'ops': []}},
        'lm_head': {}, 'embed': {}, 'norm': {}}
    types = (['linear_attention'] * 3 + ['full_attention']) * 16
    required = m.required_tensors(plan)
    counts = {}
    for name, info in required.items():
        for layer in range(len(types)):
            if info['layer_types'] and types[layer] not in info['layer_types']:
                continue
            counts[name] = counts.get(name, 0) + 1
    assert counts == {'lin.{l}': 48, 'attn.{l}': 16}
    # The ungated version, which the first implementation did, would say 64 each.
    assert counts['attn.{l}'] != 64


def test_check_rejects_a_missing_package(tmp_path):
    """A model directory without a Q4NX container must fail loudly, never pass an
    empty inventory as full coverage."""
    m = load()
    with pytest.raises((ValueError, FileNotFoundError, KeyError, RuntimeError)):
        m.check(tmp_path, tmp_path)


def test_eos_gate_requires_an_id_the_runtime_can_stop_on():
    """The server refuses a model whose tokenizer_config has no eos_token_id
    array, and silently never stops on one that names no generation id."""
    m = load()
    assert m.eos_status({"eos_token": "<|im_end|>"}, {}) == \
        dict(present=False, ids=[], generation_ids=[], agrees=False,
             resolved=[])
    ok = m.eos_status({"eos_token_id": [248046, 248044]},
                      {"eos_token_id": [248046, 248044]})
    assert ok['present'] and ok['agrees'] and ok['ids'] == [248046, 248044]
    # A single int is not the array the runtime reads.
    assert m.eos_status({"eos_token_id": 248046}, {})['present'] is False


def test_eos_gate_reports_a_generation_id_the_tokenizer_forgets():
    """Losing generation_config's second stop id is a real defect: generation
    runs past it. Report it rather than averaging it away."""
    m = load()
    status = m.eos_status({"eos_token_id": [248046]},
                          {"eos_token_id": [248046, 248044]})
    assert status['present'] is True
    assert status['agrees'] is False
    assert status['ids'] == [248046]
    assert status['generation_ids'] == [248046, 248044]
    assert 248044 in status['resolved']


def test_eos_gate_accepts_a_tokenizer_covering_every_generation_id():
    m = load()
    status = m.eos_status({"eos_token_id": [248046, 248044, 1]},
                          {"eos_token_id": [248046, 248044]})
    assert status['present'] and status['agrees']
