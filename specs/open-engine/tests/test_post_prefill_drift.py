"""Current FP32 attribution must preserve failed baseline acceptance and provenance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('post_drift',ROOT/'utilities/prefill-sequential-drift/post_targets.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


@pytest.mark.parametrize('fault',['none','unbound','incomplete','acceptance','source','measurements','preparation'])
def test_completed_failed_baseline_is_required_and_remeasured(tmp_path,monkeypatch,fault):
    m=load();prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'preparation.json').write_text('{}')
    short=tmp_path/'short';short.mkdir();(short/'report.json').write_text(json.dumps({'inputs':{}}))
    (short/'audit.json').write_text('{}')
    cases={'257':{'plain':[], 'routes':[]}};prefixes={'prefix':True}
    saved=dict(complete=True,passed=False,model_acceptance=False,sources=m.boundary.sources(),
               prepared=str(prepared),preparation_sha256=m.digest(prepared/'preparation.json'),cases=cases,prefixes=prefixes)
    if fault=='incomplete':saved['complete']=False
    elif fault=='source':saved['sources']={}
    elif fault=='measurements':saved['prefixes']={}
    elif fault=='preparation':saved['preparation_sha256']='changed'
    (tmp_path/'report.json').write_text(json.dumps(saved))
    proof=dict(complete=True,passed=fault=='acceptance',model_acceptance=False,
               report_sha256='changed' if fault=='unbound' else m.digest(tmp_path/'report.json'))
    (tmp_path/'audit.json').write_text(json.dumps(proof))
    record=dict(inputs={},baseline=str(short),baseline_hashes={n:m.digest(short/n) for n in ('report.json','audit.json')},references=str(tmp_path/'refs'))
    monkeypatch.setattr(m.boundary,'verify_preparation',lambda _: (record,{257:dict(ids_sha256='ids')},None))
    monkeypatch.setattr(m.boundary,'measure',lambda *a:(cases,prefixes))
    monkeypatch.setattr(m.boundary,'accept',lambda *a:False)
    monkeypatch.setattr(m.base,'verify',lambda _:None)
    if fault=='none':
        meta,refs,inputs=m.validate(tmp_path)
        assert refs==tmp_path/'refs/257' and str(tmp_path/'report.json') in inputs
    else:
        with pytest.raises(ValueError):m.validate(tmp_path)


def test_current_capture_layout_and_exact_saved_array_reaudit(tmp_path,monkeypatch):
    m=load();refs=tmp_path/'refs';refs.mkdir();d=tmp_path/'257/trace';d.mkdir(parents=True)
    out=tmp_path/'result';out.mkdir()
    xs=np.ones((257,2),dtype='<f4')
    xs.tofile(refs/'ref_res0.bin');xs.tofile(refs/'ref_res1.bin')
    native=np.ones((258,2),dtype='<f4')
    for route in m.drift.ROUTES:
        native.tofile(d/f'{route}_res0.bin');native.tofile(d/f'{route}_res1.bin')
    model=SimpleNamespace(matmul_w=lambda *a:None,mm=SimpleNamespace(close=lambda:None),f=SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(m.base,'spec_from_model_dir',lambda _:SimpleNamespace(hidden=2))
    monkeypatch.setattr(m.base,'Q4NX',lambda _:model)
    monkeypatch.setattr(m.drift,'replay',lambda model,spec,layer,inputs,positions:({p:inputs[p].copy() for p in positions},{p:None for p in positions}))
    monkeypatch.setattr(m.drift,'on_history',lambda model,spec,layer,x,state,p:x.copy())
    rows,artifacts,count,peak=m.evaluate(tmp_path,dict(model='model',ids=list(range(257)),hidden=2),refs,{1:(125,)},out=out)
    assert count==1 and len(rows)==2 and len(artifacts)==2
    again,_,_,_=m.evaluate(tmp_path,dict(model='model',ids=list(range(257)),hidden=2),refs,{1:(125,)},expected=out)
    assert again==rows
    p=out/'sequential_L1_P125.npz'
    with np.load(p) as saved: arrays={k:saved[k] for k in saved.files}
    arrays['native'][0]+=1;np.savez(p,**arrays)
    with pytest.raises(ValueError,match='re-audit'):
        m.evaluate(tmp_path,dict(model='model',ids=list(range(257)),hidden=2),refs,{1:(125,)},expected=out)
