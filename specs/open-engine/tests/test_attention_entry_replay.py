"""Experimental entry precision must not relabel the production ABI or inputs."""
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'utilities/attention-entry-replay'))
from entry_common import geometry, render_design, source_hash, read_candidate_norm
from recipes.load import load_spec


@pytest.fixture
def retained_recipe(monkeypatch):
    # This historical probe requires the pre-integration checkout. Model its
    # retained geometry explicitly; never reinterpret current production captures.
    import entry_common as mod
    original = mod.recipe
    def retained(s):
        r = original(s)
        if r.attn.KV_PAIR:
            r = replace(r, attn=replace(r.attn, XN_F32=False),
                        layout=replace(r.layout, AA_BYTES=266240, AA_XN=0))
        return r
    monkeypatch.setattr(mod, 'recipe', retained)


def spec():return load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')


def test_candidate_appends_fp32_norm_without_moving_production_regions(retained_recipe):
    g=geometry(spec(),'fp32')
    assert g['norm_offset']==g['production_act_bytes']
    assert g['act_bytes']==g['production_act_bytes']+20480
    assert g['norm_bytes']==20480 and g['norm_elements']==5
    b=geometry(spec(),'bf16')
    assert b['act_bytes']==b['production_act_bytes'] and b['norm_bytes']==10240


def test_render_changes_only_candidate_entry_route_and_pins_hash(retained_recipe):
    retained_source = '\n'.join([
        'HERE = Path(__file__).parent', 'OG_ELEMS = D.OG_ELEMS',
        'X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS, 2 * Q_PC + 2 * KV_PC, "attn")',
        'lno.drain(a_act, tap=bt(AA_BYTES, AA_XN, ELN), wait=True, group=tg_ln)',
    ])
    s=render_design(spec(),source=retained_source)
    assert 'AA_XN = AA_BYTES' in s and 'fp32=True' in s
    assert 'AA_XN, HID * 4)' in s and 'install_entry_worker(X)' in s
    assert 'ENTRY_SOURCE_HASH' in s
    assert source_hash(spec(),'fp32')!=source_hash(spec(),'bf16')


def test_render_rejects_upstream_source_drift(retained_recipe):
    with pytest.raises(ValueError,match='anchor'):render_design(spec(),source='different source')


def test_unsupported_geometry_or_mode():
    with pytest.raises(ValueError):geometry(spec(),'float16')
    s=load_spec(ROOT/'open_kernels/recipes/specs/qwen35-9b.json')
    with pytest.raises(ValueError):geometry(s,'fp32')


def test_read_fp32_norm_exact_low_bits_and_validation(tmp_path):
    g=dict(act_bytes=32,norm_offset=8,norm_bytes=16,hidden=4,mode='fp32')
    b=bytearray(32);x=np.full(4,1.001,np.float32);b[8:24]=x.tobytes();p=tmp_path/'act';p.write_bytes(b)
    np.testing.assert_array_equal(read_candidate_norm(p,g),x.astype(np.float64))
    p.write_bytes(b[:-1])
    with pytest.raises(ValueError):read_candidate_norm(p,g)
    b[8:12]=np.array([np.nan],np.float32).tobytes();p.write_bytes(b)
    with pytest.raises(ValueError,match='finite'):read_candidate_norm(p,g)


def test_replay_keeps_history_within_layer_and_resets_between_layers(tmp_path):
    from entry_check import replay_cfg
    m=dict(layout=dict(pool_bytes=128,hidden=2,kv_row=16,ptab_row=8),max_ctx_default=4,
           layer_types={'full_attention':{'buffers':{'consts':32,'state':{'row':16}}}})
    g=dict(act_bytes=96)
    cfg=replay_cfg(m,g,[3,7],3,tmp_path/'fixture',tmp_path/'pools',tmp_path/'out',tmp_path/'build')
    assert cfg.count('run ax pool xres consts state act ptab')==6
    assert cfg.count('load state ')==2 and cfg.count('load pool ')==2
    assert cfg.count('attnpos ax 0')==2 and cfg.count('attnpos ax 2')==2
    assert 'y_res2_t1.bin' in cfg and 'y_res6_t2.bin' in cfg
    assert 'buf act 96' in cfg and 'dump act ' in cfg


def test_pin_verification_requires_complete_exact_file_set(tmp_path):
    from entry_check import verify_inputs
    p=tmp_path/'input';p.write_bytes(b'abc')
    from entry_common import digest
    audit={'sha256':{str(p.resolve()):digest(p)}}
    verify_inputs(audit,[p])
    with pytest.raises(ValueError,match='coverage'):verify_inputs(audit,[])
    p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash'):verify_inputs(audit,[p])


def test_controlled_comparison_requires_inputs_identity_and_complete_rows():
    from entry_compare import compare_records
    base=dict(layers=[3],tokens=1,fixture_build_key='base',results=[dict(layer=3,token=0,passed=False)])
    a=dict(base,mode='bf16',control_identical=True,passed=False)
    b=dict(base,mode='fp32',passed=True,results=[dict(layer=3,token=0,passed=True)])
    assert compare_records(a,b,{'input':'hash'},{'input':'hash'})['passed']
    with pytest.raises(ValueError,match='input'):compare_records(a,b,{'input':'hash'},{'input':'other'})
    with pytest.raises(ValueError,match='control'):compare_records(dict(a,control_identical=False),b,{}, {})
    with pytest.raises(ValueError,match='coverage'):compare_records(a,dict(b,results=[]),{}, {})
    assert not compare_records(a,dict(b,passed=False,results=base['results']),{'input':'hash'},{'input':'hash'})['passed']


def test_historical_probe_rejects_already_integrated_entry():
    with pytest.raises(ValueError, match='already uses FP32'):
        geometry(spec(), 'bf16')
