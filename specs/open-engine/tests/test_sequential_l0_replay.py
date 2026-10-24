"""A long-prefix replay must retain causal state and reject inexact controls."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('native_l0', ROOT/'utilities/prefill-sequential-drift/native_l0.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def test_causal_program_initializes_state_once_and_captures_every_row(tmp_path):
    m = load()
    cfg = m.program(tmp_path, tmp_path/'kernels', dict(pool=128, consts=32, act=64, state=96, hidden=4), 243)
    lines = cfg.splitlines()
    assert lines.count('run lx pool xres consts state act') == 243
    assert sum(line.startswith('buf state ') for line in lines) == 1
    assert not any(line.startswith(('load state ', 'load act ')) for line in lines)
    for pos in range(243):
        i = lines.index(f'load xres {tmp_path}/input_{pos}.bin')
        assert lines[i+1] == 'run lx pool xres consts state act'
        assert lines[i+2:i+5] == [f'dump {k} {tmp_path}/{k}_{pos}.bin {n}'
                                   for k,n in [('xres',16),('act',64),('state',96)]]


@pytest.mark.parametrize('count',[0,244,True,2.5])
def test_program_rejects_invalid_extent(tmp_path,count):
    with pytest.raises(ValueError):
        load().program(tmp_path,tmp_path,dict(pool=128,consts=32,act=64,state=96,hidden=4),count)


def test_identity_uses_bytes_not_tolerance():
    m=load(); ref=np.ones((3,4),dtype='<f4'); got=ref.copy()
    got[1,2]=np.nextafter(got[1,2],np.float32(2))
    r=m.identity(got,ref)
    assert not r['passed'] and r['matching_positions']==2 and r['first_mismatch']==1
    assert m.identity(ref,ref)['passed']


def test_identity_distinguishes_signed_zero():
    ref=np.zeros((2,4),dtype='<f4'); got=ref.copy();got[1,0]=-0.
    assert not load().identity(got,ref)['passed']


@pytest.mark.parametrize('kind',['truncated','nan','float64'])
def test_identity_rejects_invalid_arrays(kind):
    ref=np.ones((3,4),dtype='<f4'); got=ref.copy()
    if kind=='truncated': got=got[:-1]
    if kind=='nan':got[1,2]=np.nan
    if kind=='float64':got=got.astype(np.float64)
    with pytest.raises(ValueError):load().identity(got,ref)


def test_inventory_includes_complete_state_and_activation_history():
    m=load(); names=m.output_names(243)
    assert len(names)==729 and len(set(names))==729
    assert {'xres_242.bin','state_0.bin','act_37.bin'}<=set(names)


def test_output_reader_rejects_missing_or_truncated_capture(tmp_path):
    m=load(); sizes=dict(hidden=4,act=64,state=96)
    for name in m.output_names(2):
        (tmp_path/name).write_bytes(bytes(16 if name.startswith('xres') else sizes[name.split('_')[0]]))
    assert m.read_outputs(tmp_path,sizes,2)[0].shape==(2,4)
    (tmp_path/'state_0.bin').write_bytes(bytes(95))
    with pytest.raises(ValueError):m.read_outputs(tmp_path,sizes,2)


def test_audit_retains_failed_identity_without_promoting_model(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    m=load();control=dict(passed=False,positions=243,matching_positions=242,first_mismatch=37)
    record=dict(complete=True,model_acceptance=False,preparation_sha256='hash',
                harness='runner',harness_sha256='hash',log_sha256='hash',
                identity=control,artifacts={'xres_0.bin':'capture'},control_valid=False,
                boundary_interpretation_allowed=False)
    (tmp_path/'report.json').write_text(json.dumps(record))
    monkeypatch.setattr(m,'verified',lambda out:{})
    monkeypatch.setattr(m,'digest',lambda path:'hash')
    monkeypatch.setattr(m,'score',lambda out,prepared:(control,record['artifacts']))
    assert m.audit(SimpleNamespace(out=tmp_path))==1
    result=json.loads((tmp_path/'audit.json').read_text())
    assert result['complete'] and not result['control_valid'] and not result['model_acceptance']
    record['boundary_interpretation_allowed']=True
    (tmp_path/'report.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='measurement/inventory'):m.audit(SimpleNamespace(out=tmp_path))
    assert not json.loads((tmp_path/'audit.json').read_text())['complete']


def test_run_refuses_overwriting_partial_native_capture(tmp_path):
    from types import SimpleNamespace
    (tmp_path/'state_37.bin').write_bytes(b'partial')
    with pytest.raises(ValueError,match='overwrite'):
        load().run(SimpleNamespace(out=tmp_path,harness=tmp_path/'runner'))
