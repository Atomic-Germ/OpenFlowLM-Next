"""Production-pool and activation adapters for the complete wide attention layer."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from recipes.spec import ModelSpec
from recipes import qwen35

ROOT = Path(__file__).resolve().parents[3]


def geometry(monkeypatch):
    monkeypatch.setenv('OPEN_KERNELS_UNVALIDATED', '1')
    s = ModelSpec.from_hf_config(json.loads((ROOT/'specs/open-engine/tests/fixtures/config_qwen38_27b.json').read_text()))
    return s, qwen35.layout(s, max_ctx=257)


def test_projection_adapters_keep_query_gate_and_kv_order(monkeypatch):
    from recipes.wide_attention_layer import projection_copies
    s, _ = geometry(monkeypatch)
    source = np.arange(14336, dtype=np.float32).view(np.uint8)
    dest = dict(qg=np.zeros(12288*4, np.uint8), kvn=np.zeros(2048*4, np.uint8))
    for name, dst, src, size in projection_copies(s):
        dest[name][dst:dst+size] = source[src:src+size]
    np.testing.assert_array_equal(dest['qg'].view(np.float32), np.r_[np.arange(6144), np.arange(8192,14336)])
    np.testing.assert_array_equal(dest['kvn'].view(np.float32), np.arange(6144,8192))


def test_setup_uses_full_attention_production_pool_and_norm_regions(monkeypatch):
    from recipes.wide_attention_layer import setup_commands
    s, l = geometry(monkeypatch)
    commands = setup_commands(s, l)
    plan = qwen35.pack_plan(s)['layer_types']['full_attention']
    ops = [op for op in plan['pool'] if '.self_attn.' in op['tensor']]
    offset = 0
    for op in ops[:4]:
        size = op['nch']*5120
        assert f"copy qkvgw {offset} pool {op['dst']} {size}" in commands
        offset += size
    assert offset == 14336*5120//8192*5120
    assert f'copy ow 0 pool {l.POOL_O} {5120*6144//8192*5120}' in commands
    assert f'copy lnw 0 const {l.CA_LNW} 10240' in commands
    assert f'copy postw 0 const {l.CA_POSTLN} 10240' in commands
    assert f'copy meta 0 const {l.CA_META} 2048' in commands


def test_token_program_preserves_device_cache_and_bridges_ffn_offsets(monkeypatch):
    from recipes.wide_attention_layer import token_commands
    s, l = geometry(monkeypatch)
    assert l.AA_XM != l.A_XM and l.AA_OUT2 != l.A_OUT2
    for pos in (0, 1, 255, 256):
        commands = token_commands(s, l, pos, 257)
        assert [c.split()[1] for c in commands if c.startswith('run ')] == ['ln','qkvg','attn','out','ln','ffn','ln']
        assert f'copy cache {pos*4096} new 0 4096' in commands
        assert f'copy ffnact {l.A_XM} act {l.AA_XM} 10240' in commands
        assert f'copy act {l.AA_OUT2} ffnact {l.A_OUT2} 20480' in commands
        assert all(c.split()[0] in ('run','copy') for c in commands)
        assert not any('ref' in c for c in commands)
    for pos in (-1, 257):
        with pytest.raises(ValueError, match='cache'):
            token_commands(s, l, pos, 257)


@pytest.mark.parametrize('changes', [dict(hidden=4096), dict(intermediate=16384),
    dict(num_heads=16), dict(num_kv_heads=2), dict(head_dim=128),
    dict(rotary_dim=128), dict(attn_gate=False), dict(norm_eps=1e-5)])
def test_layer_rejects_unbuilt_geometry(monkeypatch, changes):
    from recipes.wide_attention_layer import token_commands
    s, l = geometry(monkeypatch)
    with pytest.raises(ValueError, match='not implemented'):
        token_commands(replace(s, **changes), l, 0, 257)


def test_attention_artifact_rejects_wrong_dma_rows_or_stale_binary(tmp_path):
    spec = importlib.util.spec_from_file_location('attention_layer_probe',ROOT/'utilities/test-wide-attention-layer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    manifest = dict(rows=257,nh=24,kvh=4,hd=256,rot=64,sha256={})
    for name in ('final.xclbin','insts.bin'):
        (tmp_path/name).write_bytes(name.encode())
        manifest['sha256'][name] = hashlib.sha256(name.encode()).hexdigest()
    (tmp_path/'attention-fixture.json').write_text(json.dumps(manifest))
    mod.validate_attention_build(tmp_path,257)
    with pytest.raises(ValueError,match='geometry'):
        mod.validate_attention_build(tmp_path,2048)
    (tmp_path/'insts.bin').write_bytes(b'stale')
    with pytest.raises(ValueError,match='artifact'):
        mod.validate_attention_build(tmp_path,257)


def test_projection_artifact_rejects_a_different_output_size(tmp_path):
    spec = importlib.util.spec_from_file_location('attention_layer_probe',ROOT/'utilities/test-wide-attention-layer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta = dict(scope='projection',ffn=17408,projection_k=5120,projection_n=14336,weight_format='q4_1')
    path = tmp_path/'probe-toolchain.json'
    path.write_text(json.dumps(meta))
    mod.validate_projection_build(tmp_path)
    meta['projection_n'] = 16384  # The preceding DeltaNet probe would overrun this BO.
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='geometry'):
        mod.validate_projection_build(tmp_path)
