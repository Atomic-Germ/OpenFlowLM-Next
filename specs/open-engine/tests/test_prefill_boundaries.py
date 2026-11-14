"""Block partition parity cannot replace an independent numerical or cache gate."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
loader=importlib.util.spec_from_file_location('prefill_boundaries',ROOT/'utilities/prefill-boundaries/check.py')
m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)


def test_partition_covers_prompt_and_separate_continuation():
    assert m.schedule(1)==[1,1]
    assert m.schedule(255)==[255,1]
    assert m.schedule(256)==[256,1]
    assert m.schedule(257)==[256,1,1]
    with pytest.raises(ValueError):m.schedule(0)


def test_numerical_gate_rejects_nan_and_partial_shapes():
    a=np.ones(4)
    assert m.metric(a,a)['passed']
    assert not m.metric(a*2,a)['passed']
    with pytest.raises(ValueError):m.metric(a,np.ones(3))
    with pytest.raises(ValueError):m.metric(a,np.full(4,np.nan))


def test_parity_does_not_hide_wrong_reference():
    base=dict(delta=np.ones((3,2)),attn=np.ones((3,2)),conv=np.ones((3,2)),state=np.zeros((2,5,4)),kv=np.zeros((7,8),np.uint16),padding_ok=True)
    wrong=dict(base,delta=base['delta']*2)
    r=m.score_result(wrong,wrong,base)
    assert r['identical'] and not r['passed']


def test_correct_numeric_result_does_not_hide_padding_or_history_error():
    base=dict(delta=np.ones((3,2)),attn=np.ones((3,2)),conv=np.ones((3,2)),state=np.zeros((2,5,4)),kv=np.zeros((7,8),np.uint16),padding_ok=True)
    bad=dict(base,padding_ok=False)
    assert not m.score_result(bad,base,base)['passed']
    bad=dict(base,conv=base['conv'].copy());bad['conv'][0,0]=1.00001
    assert not m.score_result(bad,base,base)['passed']


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    import os
    import platform
    import shutil
    if (platform.system() != 'Linux' or platform.machine() != 'x86_64'
            or not shutil.which('g++') or 'avx2' not in Path('/proc/cpuinfo').read_text()):
        pytest.skip('native host test needs Linux x86_64, g++ and AVX2')
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
    return m.native_library(m.build(tmp_path_factory.mktemp('prefill-host')))


@pytest.mark.parametrize('prompt',[1,2,255,256,257])
def test_native_boundary_and_continuation_against_independent_reference(native,prompt):
    g=m.geometry(small=True);c=m.make_case(g,prompt+1)
    reference=m.run_reference(c,g,prompt+1)
    seq=m.run_native(native,c,g,[1]*(prompt+1),padded=False)
    blocked=m.run_native(native,c,g,m.schedule(prompt))
    assert m.score_result(blocked,seq,reference)['passed']


def test_wrong_low_cache_plane_fails_even_with_identical_native_paths():
    base=dict(delta=np.ones((3,2)),attn=np.ones((3,2)),conv=np.ones((3,2)),state=np.zeros((2,5,4)),
              kv=m.bits(np.ones((7,8))),padding_ok=True)
    broken=base['kv'].copy();broken[:,2:4]=0
    wrong=dict(base,kv=broken)
    score=m.score_result(wrong,wrong,base)
    assert score['identical'] and not score['numerical']['kv']['passed'] and not score['passed']


def test_failed_rerun_invalidates_previous_success(tmp_path,monkeypatch):
    import json
    (tmp_path/'report.json').write_text('{"passed":true}')
    monkeypatch.setattr(m.sys,'argv',['check.py','--out',str(tmp_path),'--small'])
    def failed_build(out):
        raise RuntimeError('compile failed')
    monkeypatch.setattr(m,'build',failed_build)
    with pytest.raises(RuntimeError,match='compile failed'):m.main()
    assert not json.loads((tmp_path/'report.json').read_text())['passed']
