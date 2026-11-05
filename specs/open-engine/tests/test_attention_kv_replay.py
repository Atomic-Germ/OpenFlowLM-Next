"""Widened attention KV must retain low halves and preserve the absolute gate."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[3]/'utilities/attention-kv-replay'
sys.path.insert(0,str(HERE))
import kv_common as mod
from q4nx import f32_to_bf16,bf16_to_f32
from attention_heads import compare_heads


def geom():
    return dict(tokens=2,kv_heads=2,dim=32,kw=64)


def encoded(values):
    hi=f32_to_bf16(values)
    lo=f32_to_bf16(values-bf16_to_f32(hi))
    return np.stack([hi,lo],axis=2).tobytes()


def test_kv_planes_reconstruct_history_without_swapping_k_v_or_heads(tmp_path):
    g=geom();v=(np.arange(256,dtype=np.float32).reshape(2,2,2,32)+.125)/7
    p=tmp_path/'kv';p.write_bytes(encoded(v));out=mod.read_kv(p,g,'pair')
    assert out.shape==v.shape and out.dtype==np.float64
    assert np.max(np.abs(out-v))<.0005
    assert np.max(np.abs(out-v))<np.max(np.abs(bf16_to_f32(f32_to_bf16(v))-v))/100


@pytest.mark.parametrize('kind',['short','long','nan_hi','nan_lo','infinity'])
def test_bad_kv_capture_refused(tmp_path,kind):
    a=np.zeros((2,2,2,2,32),np.uint16)
    if kind=='nan_hi':a[0,0,0,0,0]=0x7fc0
    if kind=='nan_lo':a[0,0,1,0,0]=0x7fc0
    if kind=='infinity':a[1,1,0,1,1]=0x7f80
    data=a.tobytes()
    if kind=='short':data=data[:-2]
    if kind=='long':data+=b'\0\0'
    p=tmp_path/'kv';p.write_bytes(data)
    with pytest.raises(ValueError):mod.read_kv(p,geom(),'pair')


def test_bf16_control_requires_zero_low_planes(tmp_path):
    p=tmp_path/'kv';p.write_bytes(encoded(np.full((2,2,2,32),1.003,np.float32)))
    with pytest.raises(ValueError,match='control'):mod.read_kv(p,geom(),'bf16')
    p.write_bytes(encoded(np.ones((2,2,2,32),np.float32)))
    np.testing.assert_array_equal(mod.read_kv(p,geom(),'bf16'),1)
    with pytest.raises(ValueError,match='mode'):mod.read_kv(p,geom(),'typo')


def test_high_dynamic_range_value_passes_only_with_low_half(tmp_path):
    # Position zero has probability one: output is V * sigmoid(0), independently of Q/K.
    v=np.full((2,2,2,32),100.2,np.float32);p=tmp_path/'kv';p.write_bytes(encoded(v))
    ref=v[:,1].astype(np.float64)*.5
    pair=mod.read_kv(p,geom(),'pair')[:,1]*.5
    rounded=bf16_to_f32(f32_to_bf16(v[:,1])).astype(np.float64)*.5
    assert all(r['passed'] for r in compare_heads(pair,ref))
    assert not any(r['passed'] for r in compare_heads(rounded,ref))


def test_geometry_and_mode_are_pinned_in_binary_key():
    from kv_build import source_hash
    from recipes.load import load_spec
    s=load_spec(HERE.parents[1]/'open_kernels/recipes/specs/qwen35-27b.json')
    g=mod.geometry(s,3)
    assert g['kv_row_bytes']==8192 and g['kv_format']=='bf16-hi-lo'
    assert g['input_row']==61440
    assert len({source_hash(s,'bf16',3),source_hash(s,'pair',3),source_hash(s,'pair',4)})==3


def test_failed_preparation_invalidates_stale_score(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('kv_check_test',HERE/'kv_check.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    out=tmp_path/'out';out.mkdir();(out/'score.json').write_text('{"passed":true}')
    monkeypatch.setattr('sys.argv',['check','--model-dir',str(tmp_path/'model'),'--kernel-dir',str(tmp_path/'kernels'),
        '--fixture',str(tmp_path/'fixture'),'--out',str(out),'--build',str(tmp_path/'build'),'--mode','pair'])
    with pytest.raises((ValueError,OSError)):c.main()
    assert not (out/'score.json').exists()


@pytest.mark.parametrize('damage',[None,'fixture','build_key','source_hash','geometry'])
def test_replay_source_identity_is_separate_from_retained_fixture(monkeypatch,damage):
    spec=importlib.util.spec_from_file_location('kv_check_identity',HERE/'kv_check.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    monkeypatch.setattr(c,'build_key',lambda s:'current',raising=False)
    monkeypatch.setattr(c,'source_hash',lambda s,m,t:'probe')
    g={'tokens':3};b={'production_build_key':'current','source_hash':'probe','geometry':g}
    manifest={'build_key':'retained','spec_hash':'same-spec'};fixture=dict(manifest)
    if damage=='fixture':fixture['build_key']='wrong'
    if damage=='build_key':b['production_build_key']='retained'
    if damage=='source_hash':b['source_hash']='wrong'
    if damage=='geometry':b['geometry']={'tokens':2}
    if damage is None:c.validate_sources(b,manifest,fixture,None,g,'pair')
    else:
        with pytest.raises(ValueError):c.validate_sources(b,manifest,fixture,None,g,'pair')
