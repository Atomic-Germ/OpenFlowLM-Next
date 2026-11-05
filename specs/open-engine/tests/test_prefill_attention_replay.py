"""Causal attention replay must retain per-head and independent residual gates."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load(name):
    s=importlib.util.spec_from_file_location('attention_replay_'+name,ROOT/'utilities/prefill-attention-replay'/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_build_pins_fused_attention_geometry(monkeypatch):
    monkeypatch.setenv('GQP_N','1');monkeypatch.setenv('GQP_INJECT','bad')
    e=load('build').environment()
    assert (e['GQP_N'],e['GQP_K'],e['GQP_T'],e['GQP_BREUSE'])==('14336','5120','256','0')
    assert 'GQP_INJECT' not in e


def test_build_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError):load('build').build(SimpleNamespace(out=tmp_path))


def test_small_head_failure_cannot_hide_in_global_norm():
    m=load('check');ref=np.ones((20,14336));ref[:,256:]*=1000
    candidate=ref.copy();candidate[19,0]+=0.01
    r=m.entry_scores(ref,candidate,ref)
    assert len(r['heads'])==1120 and not r['passed']
    assert sum(not h['candidate']['passed'] for h in r['heads'])==1


@pytest.mark.parametrize('bad',['nan','rows','columns'])
def test_entry_requires_full_finite_prefix(bad):
    m=load('check');x=np.ones((20,14336));y=x.copy()
    if bad=='nan':y[0,0]=np.nan
    elif bad=='rows':y=y[:19]
    else:y=y[:,:-1]
    with pytest.raises(ValueError):m.entry_scores(x,y,x)


def test_native_control_requires_each_product():
    m=load('check');x=np.ones((2,3),np.float32)
    with pytest.raises(ValueError):m.paired.require_control(x+1,x-1,x,x)


def test_independent_output_failure_is_separate():
    m=load('check');x=np.ones(4)
    r=m.output_scores(x,x,x,np.ones(4)*1.8)
    assert r['conditioned']['passed'] and not r['independent']['passed']


def test_host_shape_guard_precedes_native_call():
    m=load('check')
    with pytest.raises(ValueError):m.attention(None,np.ones((20,14335)),np.ones(256),np.ones(256),np.ones(32),[20])


def test_host_constant_shape_guard_precedes_native_call():
    m=load('check')
    with pytest.raises(ValueError):m.attention(None,np.ones((20,14336)),np.ones(255),np.ones(256),np.ones(32),[20])


def test_host_chunk_partition_guard():
    m=load('check')
    with pytest.raises(ValueError):m.attention(None,np.ones((20,14336)),np.ones(256),np.ones(256),np.ones(32),[19])


def test_reference_serialization_promotes_fp32(tmp_path):
    m=load('check');p=tmp_path/'ref.f64';x=np.ones(5120,np.float32)
    m.down.write_reference(p,x)
    assert p.stat().st_size==5120*8


def test_cache_score_reconstructs_both_planes_and_rejects_low_plane_loss():
    m=load('check');bits=np.zeros((20,4096),np.uint16)
    bits[:,:1024]=0x3f80;bits[:,1024:2048]=0x3b80
    bits[:,2048:3072]=0x4000
    planes=(bits.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    assert all(c['passed'] for c in m.cache_scores(bits,planes))
    bits[:,1024:2048]=0
    assert not all(c['passed'] for c in m.cache_scores(bits,planes))


def test_native_capture_float64_container_restores_exact_fp32():
    m=load('check');x=np.array([1.25,-3.5],np.float32)
    y=m.native_vector(x.astype(np.float64))
    assert y.dtype==np.float32 and y.tobytes()==x.tobytes()
    with pytest.raises(ValueError):m.native_vector(np.array([1.+2.**-40]))
    with pytest.raises(ValueError):m.native_vector(np.array([np.nan]))
