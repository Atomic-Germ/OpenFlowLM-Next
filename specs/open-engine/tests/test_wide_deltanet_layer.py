"""Byte-only adapters between the existing layer layout and standalone dispatches."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]


def layout(monkeypatch):
    monkeypatch.setenv('OPEN_KERNELS_UNVALIDATED','1')
    from recipes.spec import ModelSpec
    from recipes.qwen35 import layout
    spec=ModelSpec.from_hf_config(json.loads((ROOT/'specs/open-engine/tests/fixtures/config_qwen38_27b.json').read_text()))
    return spec,layout(spec)


def test_state_adapters_preserve_heads_conv_and_padding(monkeypatch):
    from recipes.wide_deltanet_layer import state_copies
    spec,l=layout(monkeypatch)
    assert l.S_ROWS==140
    compact=np.random.default_rng(48).integers(0,256,48*128*128*4,dtype=np.uint8)
    padded=np.full(l.STATE_BYTES,0xA5,np.uint8)
    for dst,src,size in state_copies(l,48,128,restore=True):
        padded[dst:dst+size]=compact[src:src+size]
    output=np.zeros_like(compact)
    for dst,src,size in state_copies(l,48,128):
        output[dst:dst+size]=padded[src:src+size]
    np.testing.assert_array_equal(compact,output)
    assert np.all(padded[:l.STATE_S_OFF]==0xA5)
    for head in range(48):
        tail=l.STATE_S_OFF+head*l.S_HEAD_BYTES+128*128*4
        assert np.all(padded[tail:tail+12*128*4]==0xA5)


def test_ab_adapter_uses_separate_bank_major_regions(monkeypatch):
    from recipes.wide_deltanet_layer import ab_copies
    spec,l=layout(monkeypatch)
    bank=5120*32*2
    raw=np.random.default_rng(32).integers(0,256,l.C_BYTES,dtype=np.uint8)
    out=np.zeros(4*bank+8192,np.uint8)
    for dst,src,size in ab_copies(l,5120,48): out[dst:dst+size]=raw[src:src+size]
    for i in range(2):
        off=i*(2*bank+4096)
        np.testing.assert_array_equal(out[off:off+bank],raw[l.C_SIDE+l.SIDE_ALPHA+i*bank:l.C_SIDE+l.SIDE_ALPHA+(i+1)*bank])
        np.testing.assert_array_equal(out[off+bank:off+2*bank],raw[l.C_SIDE+l.SIDE_BETA+i*bank:l.C_SIDE+l.SIDE_BETA+(i+1)*bank])
        np.testing.assert_array_equal(out[off+2*bank:off+2*bank+4096],raw[l.C_SIDE+l.SIDE_SMALL:l.C_SIDE+l.SIDE_SMALL+4096])


def test_projection_and_post_geometry_reject_partial_tiles():
    from recipes.wide_deltanet_layer import projection_bands,post_groups
    assert projection_bands(16384,8)==32
    assert projection_bands(5120,8)==10
    assert post_groups(48)==6 and post_groups(32)==4
    for n in (0,5130):
        with pytest.raises(ValueError): projection_bands(n,8)
    with pytest.raises(ValueError): post_groups(47)


def test_layer_program_has_no_host_math_or_state_alias(monkeypatch):
    from recipes.wide_deltanet_layer import token_commands
    s,l=layout(monkeypatch)
    cmds=token_commands(s,l)
    assert all(c.split()[0] in ('copy','run') for c in cmds)
    runs=[c for c in cmds if c.startswith('run ')]
    assert [c.split()[1] for c in runs]==['ln','qz','ab','glue','step','post','out','ln','ffn','ln']
    assert 'run step si vec so o' in runs
    assert 'run ln fo res1 ones y discard' in runs
    assert sum(c.startswith('copy si ') for c in cmds)==48
    assert sum(c.startswith('copy state ') for c in cmds)==49
    assert not any('ref' in c for c in cmds)


def test_post_metric_signed_zero_and_real_errors():
    import importlib.util
    import sys
    sys.path.insert(0,str(ROOT/'utilities'))
    module=importlib.util.spec_from_file_location('layer_probe',ROOT/'utilities/test-wide-deltanet-layer.py')
    probe=importlib.util.module_from_spec(module); module.loader.exec_module(probe)
    from ml_dtypes import bfloat16
    ref=np.zeros(4096,bfloat16)
    assert probe.post_metric(-ref,ref)['passed']
    wrong=ref.copy(); wrong[0]=1
    assert not probe.post_metric(wrong,ref)['passed']
    assert not probe.post_metric(np.full(4096,np.nan,bfloat16),ref)['passed']


@pytest.mark.parametrize('changes', [dict(lin_key_heads=8),dict(conv_kernel=3),dict(norm_eps=1e-5)])
def test_layer_program_rejects_geometry_different_from_built_kernels(monkeypatch,changes):
    from dataclasses import replace
    from recipes.wide_deltanet_layer import token_commands
    s,l=layout(monkeypatch)
    with pytest.raises(ValueError,match='not implemented'):
        token_commands(replace(s,**changes),l)
