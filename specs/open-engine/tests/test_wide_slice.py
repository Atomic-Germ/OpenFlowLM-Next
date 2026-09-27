"""Eight-layer schedule: explicit layer types and isolated persistent state."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
from recipes.spec import ModelSpec, LINEAR, FULL
from recipes import qwen35

ROOT = Path(__file__).resolve().parents[3]


def geometry(monkeypatch):
    monkeypatch.setenv('OPEN_KERNELS_UNVALIDATED','1')
    s = ModelSpec.from_hf_config(json.loads((ROOT/'specs/open-engine/tests/fixtures/config_qwen38_27b.json').read_text()))
    return s,qwen35.layout(s,max_ctx=257)


def test_prefix_uses_explicit_types_not_interval(monkeypatch):
    from recipes.wide_slice import layer_types
    s,_ = geometry(monkeypatch)
    assert layer_types(s) == (LINEAR,LINEAR,LINEAR,FULL,LINEAR,LINEAR,LINEAR,FULL)
    types = list(s.layer_types)
    types[0],types[3] = types[3],types[0]
    assert layer_types(replace(s,layer_types=tuple(types)))[0] == FULL
    with pytest.raises(ValueError,match='eight'):
        layer_types(replace(s,layer_types=(LINEAR,)*4))


def test_state_names_do_not_alias_across_layers(monkeypatch):
    from recipes.wide_slice import layer_commands
    s,l = geometry(monkeypatch)
    all_runs = []
    for i,kind in enumerate(s.layer_types[:8]):
        commands = layer_commands(s,l,i,1,257)
        assert f'load pool layer{i}/pool.bin' in commands
        assert f'load const layer{i}/const.bin' in commands
        runs = [c for c in commands if c.startswith('run ')]
        all_runs += runs
        if kind == LINEAR:
            assert f'copy d_cs 0 state{i} 0 {l.STATE_S_OFF}' in commands
            assert f'copy state{i} 0 d_conv 0 {l.STATE_S_OFF}' in commands
            assert len(runs) == 10
        else:
            assert f'run a_attn a_meta a_qg a_kvn cache{i} a_new a_og' in commands
            assert f'copy cache{i} 4096 a_new 0 4096' in commands
            assert 'copy a_meta 2048 position 0 2048' in commands
            assert len(runs) == 7
        assert not any('ref' in c or c.startswith('load x ') for c in commands)
        assert not any(f'state{j} ' in c or f'cache{j} ' in c for c in commands for j in range(8) if i!=j)
    assert len(all_runs) == 74


def test_scoping_only_renames_operands_and_preserves_offsets():
    from recipes.wide_slice import scope
    assert scope('copy act 65536 xm 0 10240','a',3) == 'copy a_act 65536 a_xm 0 10240'
    assert scope('run ln projout x postw res1 xm','d',0) == 'run ln d_projout x d_postw d_res1 d_xm'
    assert scope('run ffn pool act trace','d',4) == 'run ffn pool d_act d_trace'
    with pytest.raises(ValueError,match='command'):
        scope('load x reference.bin','a',3)


def test_invalid_prefix_or_layer_index_is_rejected(monkeypatch):
    from recipes.wide_slice import layer_types,layer_commands
    s,l = geometry(monkeypatch)
    for bad in ((LINEAR,)*8, ('unsupported',)+tuple(s.layer_types[1:8])):
        with pytest.raises(ValueError): layer_types(replace(s,layer_types=bad))
    for i in (-1,8):
        with pytest.raises(ValueError): layer_commands(s,l,i,0,257)
