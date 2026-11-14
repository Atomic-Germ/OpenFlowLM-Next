"""Signed L3 attribution must preserve real KV, exact controls and cancellation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('l3_attribution',ROOT/'utilities/prefill-sequential-drift/l3_attribution.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_paired_kv_low_planes_and_causal_prefix_identity(tmp_path):
    m=load();planes=np.zeros((3,4,1,2),np.float32)
    planes[:,0]=1;planes[:,1]=1/256;planes[:,2]=2;planes[:,3]=-1/128
    raw=m.f32_to_bf16(planes).astype('<u2').tobytes()
    for p in range(3):(tmp_path/f'state_{p}.bin').write_bytes(raw[:(p+1)*16])
    kv=m.read_native_kv(tmp_path,3,1,2)
    np.testing.assert_array_equal(kv[:,0],np.full((3,1,2),1+1/256))
    np.testing.assert_array_equal(kv[:,1],np.full((3,1,2),2-1/128))
    corrupt=bytearray((tmp_path/'state_1.bin').read_bytes());corrupt[0]^=1
    (tmp_path/'state_1.bin').write_bytes(corrupt)
    with pytest.raises(ValueError,match='prefix'):m.read_native_kv(tmp_path,3,1,2)


@pytest.mark.parametrize('fault',['size','nan'])
def test_kv_refuses_invalid_captures(tmp_path,fault):
    m=load();v=np.ones((1,4,1,2),np.float32)
    if fault=='nan':v[0,0,0,0]=np.nan
    raw=m.f32_to_bf16(v).astype('<u2').tobytes()
    (tmp_path/'state_0.bin').write_bytes(raw[:-1] if fault=='size' else raw)
    with pytest.raises(ValueError):m.read_native_kv(tmp_path,1,1,2)


def test_native_kv_attention_uses_history_gqa_and_sigmoid():
    m=load();spec=SimpleNamespace(num_heads=2,num_kv_heads=1,head_dim=2,norm_eps=1e-6,rotary_dim=0,rope_inv_freq=lambda:[])
    q=np.array([[1.,0.],[0.,1.]])
    kv=np.array([[[[1.,0.]],[[2.,4.]]],[[[0.,1.]],[[6.,8.]]]])
    got=m.native_kv_attention(q,np.zeros_like(q),np.ones(2),kv,spec,1)
    scale=1/np.sqrt(.5+1e-6)/np.sqrt(2)
    weights=np.exp([scale,0]);weights/=weights.sum()
    np.testing.assert_allclose(got[0],weights@np.array([[2.,4.],[6.,8.]])*.5)
    np.testing.assert_allclose(got[1],weights[::-1]@np.array([[2.,4.],[6.,8.]])*.5)
    with pytest.raises(ValueError):m.native_kv_attention(q,np.zeros_like(q),np.ones(2),kv,spec,0)


def test_signed_chain_closes_and_preserves_cancellation():
    m=load();nodes={k:np.array([10.,2.]) for k in m.NODES}
    nodes['entry_norm']=np.array([11.,2.])
    nodes['native']=np.array([10.1,2.])
    metrics,vectors=m.decompose(nodes)
    np.testing.assert_allclose(sum(vectors[k] for k in m.PARTS),vectors['total'])
    assert metrics['entry_norm']['maxrel']==.1 and metrics['entry_projection']['maxrel']==.1
    assert metrics['total']['maxrel']<.011
    assert metrics['local_total']==metrics['total']
    assert metrics['without_entry_norm']['maxrel']>metrics['total']['maxrel']


@pytest.mark.parametrize('fault',['missing','shape','nan'])
def test_attribution_refuses_incomplete_or_invalid_nodes(fault):
    m=load();nodes={k:np.array([1.,2.]) for k in m.NODES}
    if fault=='missing':del nodes['native']
    elif fault=='shape':nodes['native']=np.ones(3)
    else:nodes['native'][0]=np.nan
    with pytest.raises(ValueError):m.decompose(nodes)


def test_exact_control_rejects_one_ulp_and_dtype_changes():
    m=load();a=np.ones(4,dtype=np.float32);m.exact(a,a)
    b=a.copy();b[0]=np.nextafter(b[0],np.float32(2))
    with pytest.raises(ValueError):m.exact(b,a)
    with pytest.raises(ValueError):m.exact(a.astype(np.float64),a)


def test_only_complete_ordered_target_inventory_is_valid():
    m=load();rows=[dict(position=p) for p in m.TARGETS];m.coverage(rows)
    for bad in (rows[:-1],rows[::-1],rows+[rows[0]]):
        with pytest.raises(ValueError):m.coverage(bad)


@pytest.mark.parametrize('fault',['none','hash','source','incomplete','acceptance'])
def test_requires_bound_current_diagnostic_audit(fault):
    m=load();r=dict(complete=True,model_acceptance=False,sources={'source':'sha'})
    a=dict(complete=True,model_acceptance=False,report_sha256='report')
    if fault=='hash':a['report_sha256']='stale'
    if fault=='source':r['sources']={}
    if fault=='incomplete':a['complete']=False
    if fault=='acceptance':r['model_acceptance']=True
    if fault=='none':m.require_audit(r,a,'report',{'source':'sha'})
    else:
        with pytest.raises(ValueError):m.require_audit(r,a,'report',{'source':'sha'})
