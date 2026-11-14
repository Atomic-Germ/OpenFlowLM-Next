"""Full-prefix forced-tail validation must cover position zero and every overlap."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('l3_tail_sweep',ROOT/'utilities/l3-tail-replay/sweep.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_program_covers_both_full_prefixes_and_clears_position_zero(tmp_path):
    m=load();sizes=dict(pool=8,consts=8,hidden=5120,act=274432,state=2048*8192,kv_row=8192,ptab=2048*2048,ptab_row=2048)
    lines=m.program(tmp_path,tmp_path/'native',tmp_path/'build',sizes).splitlines()
    assert lines.count('run ax pool xres consts state act ptab forced')==480
    assert lines.count(f'load state {tmp_path}/zero-state.bin')==2
    assert lines.count(f'load act {tmp_path}/zero-act.bin')==2
    assert not any('_-1.bin' in s for s in lines)
    for p in range(240):assert lines.count(f'attnpos ax {p}')==2
    for p in range(1,240):assert lines.count(f'load state {tmp_path}/native/state_{p-1}.bin')==2


def test_inventory_covers_exactly_1440_raw_captures(tmp_path):
    m=load();assert len(m.names())==1440
    for n in m.names():(tmp_path/n).write_bytes(b'')
    m.inventory(tmp_path)
    (tmp_path/'control-state_0.bin').unlink()
    with pytest.raises(ValueError,match='inventory'):m.inventory(tmp_path)


def test_controls_require_all_720_original_files(tmp_path):
    m=load();capture=tmp_path/'native';capture.mkdir()
    for p in range(240):
        for k in ('act','state','xres'):
            (capture/f'{k}_{p}.bin').write_bytes(b'valid')
            (tmp_path/f'control-{k}_{p}.bin').write_bytes(b'valid')
    assert m.controls(tmp_path,capture)==720
    (tmp_path/'control-xres_0.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='control'):m.controls(tmp_path,capture)


def test_summary_retains_regressions_even_if_tolerances_pass():
    m=load();rows=[]
    for p in range(240):
        c=dict(control=dict(maxabs=2.,maxrel=.001,passed=True),candidate=dict(maxabs=1.,maxrel=.0005,passed=True),improved=True)
        rows.append(dict(position=p,passed=True,primary=c,model_reference=c))
    rows[17]['primary']=dict(control=dict(maxabs=1.,maxrel=.0005,passed=True),candidate=dict(maxabs=2.,maxrel=.001,passed=True),improved=False)
    r=m.summarize(rows)
    assert r['passed']==240 and r['primary_improved']==239 and r['primary_regressions']==[17]
    with pytest.raises(ValueError):m.summarize(rows[:-1])
    with pytest.raises(ValueError):m.summarize(rows[::-1])


def test_selected_array_overlap_is_byte_exact_including_signed_zero(tmp_path):
    m=load();a=dict(x=np.array([0.,1.],dtype='<f4'));np.savez(tmp_path/'P125.npz',**a)
    assert m.overlap_arrays(a,tmp_path/'P125.npz')==1
    a['x'][0]=-0.0
    with pytest.raises(ValueError,match='overlap'):m.overlap_arrays(a,tmp_path/'P125.npz')


def test_partial_run_is_not_overwritten(tmp_path):
    m=load();(tmp_path/'candidate-act_0.bin').write_bytes(b'partial')
    with pytest.raises(ValueError,match='overwrite'):m.run(SimpleNamespace(out=tmp_path))


def test_audit_does_not_promote_failed_sweep(tmp_path,monkeypatch):
    m=load();result=dict(candidate_passed=False,exact_controls=720,artifacts={},summary={'passed':239},overlap_arrays=39)
    r=dict(complete=True,model_acceptance=False,preparation_sha256='hash',harness='runner',harness_sha256='hash',harness_evidence_sha256='hash',log_sha256='hash',result=result)
    (tmp_path/'report.json').write_text(json.dumps(r))
    monkeypatch.setattr(m,'verified',lambda p:{})
    monkeypatch.setattr(m,'digest',lambda p:'hash')
    monkeypatch.setattr(m.native,'verify_harness',lambda p:'hash')
    monkeypatch.setattr(m,'evaluate',lambda *a:result)
    assert m.audit(SimpleNamespace(out=tmp_path))==1
    proof=json.loads((tmp_path/'audit.json').read_text())
    assert proof['complete'] and not proof['candidate_passed'] and not proof['model_acceptance']
