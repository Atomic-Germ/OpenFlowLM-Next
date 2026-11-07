"""FP32 candidate acceptance requires exact causal controls and complete coverage."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('post_replay',ROOT/'utilities/deltanet-post-fp32/replay.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_config_keeps_one_state_and_reads_original_inputs(tmp_path):
    m=load();size=dict(pool=128,consts=64,hidden=4,act=80,state=32)
    code=m.program(tmp_path,tmp_path/'capture',tmp_path/'build',size)
    assert code.count('run lx pool xres consts state act')==243
    assert 'load state ' not in code and 'load act ' not in code
    assert f'buf act 80' in code and f'load xres {tmp_path}/capture/input_242.bin' in code
    assert f'xclbin lx {tmp_path}/build/final.xclbin' in code


def test_region_identity_requires_actual_byte_equality():
    m=load();a=np.ones(4,dtype='<f4');b=a.copy();b[2]=np.nextafter(b[2],np.float32(2))
    assert m.equal_region(a.tobytes(),a.tobytes(),0,16)
    assert not m.equal_region(a.tobytes(),b.tobytes(),0,16)
    with pytest.raises(ValueError):m.equal_region(a.tobytes(),b.tobytes(),4,16)


def test_aggregate_requires_all_controls_and_residuals():
    m=load();rows=[dict(position=p,upstream_exact=True,residual=dict(passed=True),control_exact=True) for p in range(243)]
    assert m.accept('control',rows)
    assert m.accept('fp32',rows)
    rows[37]['upstream_exact']=False
    assert not m.accept('fp32',rows)
    rows[37]['upstream_exact']=True;rows[241]['residual']['passed']=False
    assert not m.accept('fp32',rows)
    rows[241]['control_exact']=False
    assert not m.accept('control',rows)
    with pytest.raises(ValueError):m.accept('fp32',rows[:-1])
    with pytest.raises(ValueError):m.accept('bogus',rows)
