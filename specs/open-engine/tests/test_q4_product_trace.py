"""Product capture diagnoses operands separately from local accumulation."""
import sys
import json
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'utilities'))


def test_product_records_restore_lane_order():
    from wide_down_blocks import product_trace
    raw=np.arange(16*2560,dtype=np.float32).reshape(16,2560)
    got=product_trace(raw.ravel(),53)
    assert got.shape==(9,5,32)
    assert got[2,3,9]==raw[6,1120+(2*5+3)*32+20]
    with pytest.raises(ValueError): product_trace(raw.ravel(),128)


def test_product_diagnosis_separates_operand_and_compensation_errors():
    from wide_down_blocks import product_diagnosis
    t=np.zeros((9,5,32),np.float32)
    terms=np.array([2**24,1,-2**24,3,0,0,0,0,0],np.float32)
    t[:,0,:]=terms[:,None];t[:,1,:]=1;t[:,2,:]=terms[:,None]
    exact=terms.astype(np.float64).cumsum()
    t[:,3,:]=exact.astype(np.float32)[:,None]
    t[:,4,:]=(exact-exact.astype(np.float32))[:,None]
    r=product_diagnosis(t,4,0)
    assert r['operand_error']==0 and r['product_errors']==0
    assert all(x==0 for x in r['accumulation_error'])
    t[1,4,:]=0
    r=product_diagnosis(t,4,0)
    assert r['accumulation_error'][1]==-1 and r['operand_error']==0
    assert product_diagnosis(t,5,0)['operand_error']==-1


def test_real_cancellation_fixture_needs_exact_addition_not_more_tail_terms():
    data=json.loads((Path(__file__).parent/'fixtures/q4_product_cancellation.json').read_text())
    for case in data['cases']:
        a,b=case['high_before'],case['term']
        expected=np.float32(a+b)
        assert float(expected)==case['ieee_high']
        # The selected subtraction itself is exactly representable in FP32;
        # ordinary rounding error cannot explain the measured deviation.
        assert float(expected)==a+b
        assert case['device_high']!=float(expected)
        assert case['device_low']==case['low_before']
        assert (case['device_high']+case['device_low'])-(a+b+case['low_before'])==case['error']
