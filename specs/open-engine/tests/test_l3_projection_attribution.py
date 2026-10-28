"""Projection error attribution must close and require audited native controls."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('l3_projection_attribution',ROOT/'utilities/l3-projection-replay/attribute.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_decode_actual_integer_table_and_both_sum_planes():
    m=load();xi=np.arange(-16,16,dtype='<i2');shift=np.array([4],dtype='<i4')
    hi=np.array([0x3f80],dtype='<u2');lo=np.array([0x3b80],dtype='<u2')
    x,sums=m.table(xi.tobytes()+shift.tobytes()+hi.tobytes()+lo.tobytes(),32)
    np.testing.assert_array_equal(x,xi.astype(float)/16)
    np.testing.assert_array_equal(sums,[1+1/256])


@pytest.mark.parametrize('fault',['size','shift','nan'])
def test_invalid_table_fails(fault):
    m=load();data=bytearray(72)
    if fault=='size':data+=b'\x00'
    if fault=='shift':data[64:68]=np.array([999],dtype='<i4').tobytes()
    if fault=='nan':data[68:70]=np.array([0x7fc0],dtype='<u2').tobytes()
    with pytest.raises(ValueError):m.table(bytes(data),32)


def test_signed_cancellation_is_preserved():
    m=load();ref=np.array([10.,-20.]);quant=ref+[2,-1];native=quant+[-1.5,.5]
    metrics,v=m.decompose(ref,quant,native)
    np.testing.assert_array_equal(v['activation_quantization'],[2,-1])
    np.testing.assert_array_equal(v['arithmetic_remainder'],[-1.5,.5])
    np.testing.assert_array_equal(v['total'],v['activation_quantization']+v['arithmetic_remainder'])
    assert metrics['without_quantization']['maxrel']>metrics['total']['maxrel']
    assert all(x['reference_maxabs']==20 for x in metrics.values())


@pytest.mark.parametrize('fault',['empty','shape','nan'])
def test_invalid_vectors_fail(fault):
    m=load();a=np.ones(4);b=a.copy()
    if fault=='empty':a=b=np.empty(0)
    if fault=='shape':b=b[:-1]
    if fault=='nan':b[0]=np.nan
    with pytest.raises(ValueError):m.decompose(a,b,a)


def test_packed_decoder_checks_band_order():
    m=load();chunks=np.zeros((4,5120),dtype=np.uint8)
    # Two row tiles by two K tiles. Constants differ in every raster tile.
    for i in range(4):chunks[i,512:1024].view('<u2')[:]=np.float32(i+1).view(np.uint32)>>16
    from recipes.pack import std_perm
    w=m.weights(chunks[std_perm(4,512)].tobytes(),64,512)
    assert w.dtype==np.float64 and w.shape==(64,512)
    for row in range(64):
        for col in (0,255,256,511):assert w[row,col]==1+2*(row//32)+col//256


def test_requires_complete_bound_successful_audit():
    m=load();summary=dict(control_valid=True,exact_roles=960,total_roles=960,
        exact_tables=240,total_tables=240,table_valid=True)
    r=dict(complete=True,model_acceptance=False,interpretation_allowed=True,summary=summary)
    a=dict(r,report_sha256='hash',artifacts=480)
    m.require_control(r,a,'hash')
    for key,value in [('complete',False),('model_acceptance',True),('interpretation_allowed',False),('report_sha256','other'),('artifacts',479)]:
        bad=dict(a);bad[key]=value
        with pytest.raises(ValueError):m.require_control(r,bad,'hash')


def test_target_inventory_requires_every_position_and_role():
    m=load();rows=[dict(position=p,role=k) for k in ('q','gate','k','v') for p in range(240)]
    m.coverage(rows)
    with pytest.raises(ValueError):m.coverage(rows[:-1])
    rows[0],rows[1]=rows[1],rows[0]
    with pytest.raises(ValueError):m.coverage(rows)
