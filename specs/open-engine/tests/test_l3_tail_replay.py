"""A forced-OG full-layer replay must retain exact controls and unchanged gates."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load(name='replay'):
    s=importlib.util.spec_from_file_location('l3_tail_'+name,ROOT/f'utilities/l3-tail-replay/{name}.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_generated_design_changes_only_attention_output_source():
    m=load('build');s=m.render()
    assert 'ptab: In, forced: In, *,' in s
    assert 'px.fill(x_prod, a_forced, bt(QW * 4, 0, OG_ELEMS * ELEM))' in s
    assert 'X.ffn_body(win, xin, yout, B, K)' in s
    assert 'XL.attn_workers(of_ain, of_aout, of_og, a, afns)' in s


def test_design_refuses_changed_anchor():
    m=load('build')
    with pytest.raises(ValueError,match='anchor'):m.render('not the current ax source')


def test_forced_input_keeps_exact_fp32_output():
    m=load();og=np.arange(6144,dtype='<f4')
    assert m.forced_input(og)==og.tobytes()
    with pytest.raises(ValueError):m.forced_input(og.astype(float))
    with pytest.raises(ValueError):m.forced_input(og[:-1])
    og[0]=np.nan
    with pytest.raises(ValueError):m.forced_input(og)


def test_control_requires_all_three_full_native_captures(tmp_path):
    m=load();capture=tmp_path/'native';capture.mkdir()
    for p in m.TARGETS:
        for kind in ('act','state','xres'):
            (capture/f'{kind}_{p}.bin').write_bytes(b'original')
            (tmp_path/f'control-{kind}_{p}.bin').write_bytes(b'original')
    assert m.controls(tmp_path,capture)==9
    (tmp_path/'control-act_239.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='control'):m.controls(tmp_path,capture)


def test_program_restores_causal_history_and_original_scratch_each_time(tmp_path):
    m=load();s=dict(pool=8,consts=8,hidden=5120,act=8,state=4096*8192,kv_row=8192,ptab=4096*2048,ptab_row=2048)
    lines=m.program(tmp_path,tmp_path/'source',tmp_path/'build',s).splitlines()
    assert lines.count('run ax pool xres consts state act ptab forced')==6
    for p in m.TARGETS:
        assert lines.count(f'load state {tmp_path}/source/state_{p-1}.bin')==2
        assert lines.count(f'load act {tmp_path}/source/act_{p-1}.bin')==2
        assert lines.count(f'attnpos ax {p}')==2


def test_primary_comparison_reports_regression_separately_from_pass():
    m=load();ref=np.ones(8);r=m.compare(ref+1e-5,ref+2e-5,ref)
    assert r['candidate']['passed'] and not r['improved']


def test_capture_inventory_rejects_missing_and_extra(tmp_path):
    m=load()
    for n in m.names():(tmp_path/n).write_bytes(b'')
    m.inventory(tmp_path)
    (tmp_path/'candidate-act_0.bin').write_bytes(b'')
    with pytest.raises(ValueError,match='inventory'):m.inventory(tmp_path)


def test_partial_native_run_cannot_be_overwritten(tmp_path):
    m=load();(tmp_path/'candidate-act_239.bin').write_bytes(b'partial')
    with pytest.raises(ValueError,match='overwrite'):m.run(SimpleNamespace(out=tmp_path))


def test_position_record_argument_is_not_reused_for_forced_output():
    m=load('build');s=m.render()
    assert 'PTAB_BYTES +=' not in s
    assert 'pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, PTAB_ROW, PTAB_ROW))' in s
    assert 'ptab_ty, np.ndarray[(QW * 4,), np.dtype[np.uint8]]' in s


def test_audit_preserves_failed_numerics(tmp_path,monkeypatch):
    import json
    m=load();result=dict(candidate_passed=False,exact_controls=9,artifacts={})
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
