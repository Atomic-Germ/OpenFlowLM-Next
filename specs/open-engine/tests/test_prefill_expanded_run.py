"""Expanded resident validation must bind controls, provenance and independent gates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load(name='run'):
    s=importlib.util.spec_from_file_location('expanded_validation_'+name,ROOT/'utilities/prefill-expanded-cli'/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_preparation_requires_exact_build_inventory():
    m=load();r=dict(complete=True,manifest_matches_current=True,builds={k:{} for k in ('combined','expanded','expanded-repeat')})
    m.check_preparation(r)
    for names in (('expanded',),('paired','weighted','combined'),('combined','expanded','expanded-repeat','extra')):
        with pytest.raises(ValueError):m.check_preparation(r|{'builds':{k:{} for k in names}})


def test_missing_preparation_preserves_failure(tmp_path):
    m=load();a=SimpleNamespace(out=tmp_path/'out',prepared=tmp_path/'missing')
    with pytest.raises(FileNotFoundError):m.run(a)
    r=json.loads((a.out/'report.json').read_text())
    assert not r['complete'] and not r['passed'] and not r['model_acceptance']
    with pytest.raises(FileExistsError):m.run(a)


def setup_run(m,monkeypatch,tmp_path):
    refs=tmp_path/'prep';refs.mkdir();(refs/'preparation.json').write_text('{}')
    record={'baseline':str(tmp_path/'old'),'builds':{k:{'directory':str(tmp_path/k),'cli_sha256':k} for k in m.VARIANTS}}
    monkeypatch.setattr(m,'preparation',lambda p:(record,dict(layers=64,ids=[1],hidden=5120),None))
    monkeypatch.setattr(m,'device_lock',lambda:nullcontext())
    monkeypatch.setattr(m.base,'memory_preflight',lambda n:40*2**30)
    return SimpleNamespace(out=tmp_path/'run',prepared=refs)


def test_control_mismatch_stops_before_candidate(tmp_path,monkeypatch):
    m=load();a=setup_run(m,monkeypatch,tmp_path);called=[]
    monkeypatch.setattr(m,'capture_variant',lambda variant,*args:called.append(variant) or {'passed':False})
    monkeypatch.setattr(m,'require_identity',lambda *args,**kw:(_ for _ in ()).throw(ValueError('control mismatch')))
    with pytest.raises(ValueError,match='control mismatch'):m.run(a)
    assert called==['combined']
    assert not json.loads((a.out/'report.json').read_text())['complete']


def test_low_memory_prevents_any_native_capture(tmp_path,monkeypatch):
    m=load();a=setup_run(m,monkeypatch,tmp_path);called=[]
    monkeypatch.setattr(m.base,'memory_preflight',lambda n:(_ for _ in ()).throw(RuntimeError('low RAM')))
    monkeypatch.setattr(m,'capture_variant',lambda *args:called.append(args))
    with pytest.raises(RuntimeError,match='low RAM'):m.run(a)
    assert not called


def test_numerical_failure_is_complete_not_model_acceptance(tmp_path,monkeypatch):
    m=load();a=setup_run(m,monkeypatch,tmp_path)
    monkeypatch.setattr(m,'capture_variant',lambda *args:{'passed':False})
    monkeypatch.setattr(m,'require_identity',lambda *args,**kw:{'capture':True})
    assert m.run(a)==1
    r=json.loads((a.out/'report.json').read_text())
    assert r['complete'] and not r['passed'] and not r['model_acceptance']


def test_repeat_identity_cannot_hide_accuracy_failure():
    m=load('repeat');rows=[{'route':r,'first':{'passed':False},'second':{'passed':False},'repeat_identity':{'x':True}} for r in m.runner.ROUTES]
    assert m.aggregate(rows)==dict(passed=True,numerical_passed=False)
    rows[1]['repeat_identity']['x']=False
    with pytest.raises(ValueError):m.aggregate(rows)


def test_repeat_requires_all_routes_and_both_requests():
    m=load('repeat');rows=[{'route':r,'first':{'passed':True},'second':{'passed':True},'repeat_identity':{'x':True}} for r in m.runner.ROUTES]
    with pytest.raises(ValueError):m.aggregate(rows[:2])
    rows[0]['first']['passed']=False
    assert not m.aggregate(rows)['numerical_passed']
