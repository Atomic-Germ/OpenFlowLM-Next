"""Real RMSNorm replay distinguishes local rounding from propagated error."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from ml_dtypes import bfloat16

ROOT=Path(__file__).resolve().parents[3]


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'utilities'))
    spec=importlib.util.spec_from_file_location('ln_diagnosis',ROOT/'utilities/diagnose-wide-ln.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_strict_norm_replay_rejects_one_rounding_error(monkeypatch,tmp_path):
    m=module(monkeypatch)
    x=np.linspace(-1,1,5120,dtype=np.float32);w=np.ones(5120,bfloat16)
    expected=m.reference(x,w)
    m.write_fixture(tmp_path,x,w,expected,{'final.xclbin':b'kernel','insts.bin':b'inst'},trace=False,index=786)
    for i in range(2):
        (tmp_path/f'y{i}.bin').write_bytes(x.tobytes()+m.GUARD)
        (tmp_path/f'xn{i}.bin').write_bytes(expected.tobytes()+m.GUARD)
    assert m.compare(tmp_path)==0
    wrong=expected.copy();wrong.view(np.uint16)[786]+=1
    for i in range(2):(tmp_path/f'xn{i}.bin').write_bytes(wrong.tobytes()+m.GUARD)
    assert m.compare(tmp_path)==1
    (tmp_path/'w.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):m.compare(tmp_path)


def test_ln_reference_rounds_fp32_before_bf16(monkeypatch):
    m=module(monkeypatch)
    x=np.linspace(.01,.3,5120,dtype=np.float32);w=np.ones(5120,bfloat16)
    expected=(x.astype(np.float64)/np.sqrt(np.mean(x.astype(np.float64)**2)+1e-6)).astype(np.float32).astype(bfloat16)
    np.testing.assert_array_equal(m.reference(x,w),expected)
