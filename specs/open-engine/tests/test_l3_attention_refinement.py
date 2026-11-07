"""Native attention controls must precede interpretation of refined projections."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('l3_attention_replay', ROOT/'utilities/l3-attention-replay/replay.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def test_schedule_has_no_future_rows_and_initial_dummy_is_masked():
    m = load()
    rows = m.schedule(240)
    assert len(rows) == 240
    for p, r in enumerate(rows):
        assert r == dict(position=p, cache_offset=p*8192, cache_bytes=max(1,p)*8192,
                         input_offset=p*61440, output_offset=p*24576)


@pytest.mark.parametrize('n', [0, 241, True, 1.5])
def test_schedule_rejects_invalid_extent(n):
    with pytest.raises(ValueError): load().schedule(n)


def test_pack_preserves_fp32_projection_bytes_and_metadata():
    m=load(); q=np.arange(6144,dtype='<f4').reshape(24,256)
    projections=dict(q=q,gate=-q,k=q[:4],v=-q[:4])
    meta=bytes(2048);ptab=bytearray(2048)
    np.frombuffer(ptab,'<i4')[:3]=[239,239,0]
    data=m.pack(projections,meta,bytes(ptab),239)
    assert data==meta+bytes(ptab)+q.tobytes()+q[:4].tobytes()+(-q[:4]).tobytes()+(-q).tobytes()
    assert len(data)==61440
    with pytest.raises(ValueError): m.pack(projections,meta,bytes(ptab),238)
    projections['q']=q.astype(float)
    with pytest.raises(ValueError): m.pack(projections,meta,bytes(ptab),239)


def test_controls_require_every_output_and_entire_native_kv_prefix(tmp_path):
    m=load();out=np.zeros((240,24,256),dtype='<f4');kv=bytes(240*8192)
    for p in range(240): (tmp_path/f'state_{p}.bin').write_bytes(kv[:(p+1)*8192])
    assert m.controls(out,out.copy(),kv,tmp_path)==dict(outputs=240,kv_prefixes=240)
    changed=out.copy();changed[239,0,0]=-0.0
    with pytest.raises(ValueError,match='control'):m.controls(changed,out,kv,tmp_path)
    (tmp_path/'state_125.bin').write_bytes(b'X'+kv[1:126*8192])
    with pytest.raises(ValueError,match='control'):m.controls(out,out,kv,tmp_path)


def test_program_zeroes_cache_and_poisons_outputs_for_both_trajectories(tmp_path):
    m=load();lines=m.program(tmp_path,tmp_path/'build').splitlines()
    assert lines.count(f'load kv {tmp_path}/zero.bin')==2
    assert lines.count(f'load og {tmp_path}/poison.bin')==2
    assert lines.count('run attention input kv og')==2
    assert lines.index(f'dump og {tmp_path}/control-og.bin 5898240')<lines.index(f'load input {tmp_path}/candidate-input.bin')


def test_head_summary_preserves_failures_and_selected_regressions():
    m=load();ref=np.ones((240,24,256));control=ref.copy();candidate=ref.copy()
    candidate[239,0,0]+=0.03
    r=m.measure(control,candidate,ref,ref)
    assert not r['candidate_passed']
    assert r['summary']['candidate_primary']['passed']==5759
    assert r['selected'][-1]['position']==239 and not r['selected'][-1]['improved']


def test_bad_capture_does_not_pass_finite_check(tmp_path):
    m=load();p=tmp_path/'og.bin';np.full(240*6144,np.nan,dtype='<f4').tofile(p)
    with pytest.raises(ValueError):m.read_output(p)


def test_position_zero_requires_dummy_transfer_with_no_valid_cache_row():
    m=load();data={k:np.zeros((n,256),'<f4') for k,n in [('q',24),('gate',24),('k',4),('v',4)]}
    ptab=bytearray(2048)
    with pytest.raises(ValueError):m.pack(data,bytes(2048),bytes(ptab),0)
    np.frombuffer(ptab,'<i4')[1]=1
    assert len(m.pack(data,bytes(2048),bytes(ptab),0))==61440


def test_partial_run_cannot_be_overwritten(tmp_path):
    m=load();(tmp_path/'control-kv.bin').write_bytes(b'partial')
    with pytest.raises(ValueError,match='overwrite'):m.run(SimpleNamespace(out=tmp_path))


def test_audit_cannot_turn_failed_numerics_into_success(tmp_path,monkeypatch):
    m=load();result=dict(candidate_passed=False,controls=dict(outputs=240,kv_prefixes=240),artifacts={})
    r=dict(complete=True,model_acceptance=False,preparation_sha256='hash',harness='runner',
           harness_sha256='hash',harness_evidence_sha256='hash',log_sha256='hash',result=result)
    (tmp_path/'report.json').write_text(json.dumps(r))
    monkeypatch.setattr(m,'verified',lambda p:{})
    monkeypatch.setattr(m,'digest',lambda p:'hash')
    monkeypatch.setattr(m.native,'verify_harness',lambda p:'hash')
    monkeypatch.setattr(m,'evaluate',lambda *a:result)
    assert m.audit(SimpleNamespace(out=tmp_path))==1
    proof=json.loads((tmp_path/'audit.json').read_text())
    assert proof['complete'] and not proof['candidate_passed'] and not proof['model_acceptance']
    r['log_sha256']='changed';(tmp_path/'report.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='unbound'):m.audit(SimpleNamespace(out=tmp_path))
