"""Signed attribution must preserve cancellation, reference identity and coverage."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('post_attribution',ROOT/'utilities/prefill-sequential-drift/post_attribution.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_signed_chain_preserves_cancellation_and_common_scale():
    m=load();nodes={name:np.array([10.,-5.]) for name in m.NODES}
    nodes['post_rounded']=np.array([11.,-7.]);nodes['post_native']=np.array([10.5,-5.5])
    nodes['native']=np.array([10.1,-5.2])
    metrics,v=m.decompose(nodes)
    np.testing.assert_allclose(sum(v[k] for k in m.PARTS),v['total'],atol=1e-14)
    assert v['post_rounding'][1]==-2 and v['post_arithmetic'][1]==1.5
    assert metrics['post_rounding']['maxrel']==.2
    assert metrics['post_arithmetic']['maxrel']==.15
    np.testing.assert_array_equal(v['without_post_rounding'],nodes['native']-v['post_rounding'])


@pytest.mark.parametrize('kind',['missing','shape','nan','inf'])
def test_invalid_chain_is_rejected(kind):
    m=load();nodes={k:np.ones(3) for k in m.NODES}
    if kind=='missing':del nodes['z']
    if kind=='shape':nodes['z']=np.ones(2)
    if kind=='nan':nodes['z'][0]=np.nan
    if kind=='inf':nodes['z'][0]=np.inf
    with pytest.raises(ValueError):m.decompose(nodes)


def test_reference_gate_checks_every_bit():
    m=load();ref=np.array([1.,0.],dtype='<f4')
    m.exact_reference(ref.astype(np.float64),ref)
    changed=ref.copy();changed[0]=np.nextafter(changed[0],np.float32(2))
    with pytest.raises(ValueError):m.exact_reference(changed,ref)
    changed=ref.copy();changed[1]=-0.
    with pytest.raises(ValueError):m.exact_reference(changed,ref)


def test_summary_rejects_missing_duplicate_or_wrong_targets():
    m=load();rows=[dict(position=p) for p in m.TARGETS]
    m.check_coverage(rows)
    for bad in (rows[:-1],rows+[rows[0]],[dict(position=99),*rows[1:]]):
        with pytest.raises(ValueError):m.check_coverage(bad)


def test_zero_reference_does_not_hide_error():
    m=load();nodes={k:np.zeros(2) for k in m.NODES};nodes['native'][0]=1
    metrics,_=m.decompose(nodes)
    assert not metrics['total']['passed'] and metrics['total']['maxrel'] is None


def test_fp64_tail_uses_nonlinear_ffn():
    m=load();w=dict(post_norm=np.ones(2),up=np.eye(2),gate=np.eye(2),down=np.eye(2))
    res=np.array([1.,2.]);norm=res/np.sqrt(np.mean(res**2)+1e-6)
    expected=res+norm*norm/(1+np.exp(-norm))
    np.testing.assert_allclose(m.tail(res,w,1e-6),expected,rtol=1e-14)
