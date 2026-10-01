"""Separate projection errors from attention math using fixed captured inputs."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from ml_dtypes import bfloat16

ROOT=Path(__file__).resolve().parents[3]


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'utilities'))
    spec=importlib.util.spec_from_file_location('attention_diagnosis',ROOT/'utilities/diagnose-wide-attention.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_single_row_separates_local_math_from_projected_value(monkeypatch):
    m=module(monkeypatch)
    qg=np.zeros((2,24,256),np.float32)
    kv=np.zeros((2,4,256),np.float32);kv[1]=2
    norms=np.ones((2,256),bfloat16);cs=np.r_[np.ones(32),np.zeros(32)].astype(np.float32)
    cache=np.full((2,2,4,256),np.nan,bfloat16)
    new,og=m.conditional(qg,kv,norms,cs,cache,0)
    np.testing.assert_array_equal(og,np.ones((24,256),bfloat16))
    assert new.shape==(2,4,256)
    got=og.copy();got[0,0]=1.0078125
    report=m.differences(got,og,np.full_like(og,.5))
    assert report['local']==1 and report['propagated']==6144
    assert report['device_vs_reference']==6144


def test_capture_rejects_guard_nonfinite_and_size(monkeypatch,tmp_path):
    m=module(monkeypatch);p=tmp_path/'input.bin'
    for raw in (np.array([1],np.float32).tobytes()+bytes(64),
                np.array([np.nan],np.float32).tobytes()+m.GUARD,b''):
        p.write_bytes(raw)
        with pytest.raises(ValueError):m.capture(p,np.float32,(1,))


def test_real_replay_roundtrips_reference_and_checks_repeat_and_hash(monkeypatch,tmp_path):
    import json
    monkeypatch.setenv('OPEN_KERNELS_UNVALIDATED','1')
    m=module(monkeypatch);source=tmp_path/'source';kernel=tmp_path/'kernel';out=tmp_path/'replay'
    source.mkdir();kernel.mkdir();(source/'layer3').mkdir()
    tag='cold-0-layer3';rows=8
    for name in ('final.xclbin','insts.bin'):(kernel/name).write_bytes(b'kernel')
    (kernel/'attention-fixture.json').write_text(json.dumps(dict(nh=24,kvh=4,hd=256,rot=64,rows=rows,
        sha256={n:m.sha(kernel/n) for n in ('final.xclbin','insts.bin')})))
    (kernel/'attention-results.json').write_text(json.dumps(dict(passed=True)))
    const=np.zeros(24576,np.uint8);const[20480:21504]=np.ones(512,bfloat16).view(np.uint8)
    const.tofile(source/'layer3/const.bin')
    pos=np.zeros(2048,np.uint8);pos[:8]=np.array([0,rows],np.int32).view(np.uint8)
    pos[512:768]=np.r_[np.ones(32),np.zeros(32)].astype(np.float32).view(np.uint8)
    pos.tofile(source/'cold-0-position.bin')
    np.savez(source/f'{tag}-ref.npz',new=np.zeros((2,4,256),np.float32),og=np.zeros(6144,np.float32))
    for name,shape,dtype in [('qg',(2,24,256),np.float32),('kvn',(2,4,256),np.float32),
                             ('new',(2,4,256),bfloat16),('og',(24,256),bfloat16)]:
        (source/f'{tag}-got-{name}.bin').write_bytes(np.zeros(shape,dtype).tobytes()+m.GUARD)
    (source/f'{tag}-input-state.bin').write_bytes(np.full((rows,2,4,256),np.nan,bfloat16).tobytes()+m.GUARD)
    names=['layer3/const.bin','cold-0-position.bin',f'{tag}-ref.npz']
    (source/'slice-fixture.json').write_text(json.dumps(dict(rows=rows,kernels={},
        fixtures={n:m.sha(source/n) for n in names},cases=[dict(tag=tag,kind='full_attention',layer=3,token='cold-0',pos=0)])))
    m.prepare(source,tag,kernel,out)
    for n in ('new','og'):
        for i in range(2):(out/f'{n}{i}.bin').write_bytes((source/f'{tag}-got-{n}.bin').read_bytes())
    assert m.compare(out,True)==0
    (out/'og1.bin').write_bytes(np.ones((24,256),bfloat16).tobytes()+m.GUARD)
    assert m.compare(out,True)==1
    (out/'reference.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='fixture changed'):m.compare(out,True)
