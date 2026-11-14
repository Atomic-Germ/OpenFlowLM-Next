"""Boundary acceptance requires every length, route and causal prefix."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location('post_boundaries', ROOT/'utilities/prefill-production/post_boundaries.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passing(m):
    return {str(n): dict(plain=[dict(route=r, passed=True) for r in m.post.old.runner.ROUTES],
                         routes=[dict(route=r, first=dict(passed=True), identity={'file':True})
                                 for r in m.post.old.runner.ROUTES], schedule_identity={'file':True})
            for n in m.LENGTHS}


@pytest.mark.parametrize('fault', ['none','numerical','identity','schedule','prefix','missing_length','missing_route','empty_prefix'])
def test_boundary_gate_cannot_hide_partial_results(fault):
    m=load(); cases=passing(m); prefixes={'prefix':True}
    if fault=='numerical': cases['257']['routes'][0]['first']['passed']=False
    elif fault=='identity': cases['256']['routes'][0]['identity']['file']=False
    elif fault=='schedule': cases['255']['schedule_identity']['file']=False
    elif fault=='prefix': prefixes['prefix']=False
    elif fault=='missing_length': cases.pop('256')
    elif fault=='missing_route': cases['257']['plain'].pop()
    elif fault=='empty_prefix': prefixes={}
    if fault in ('missing_length','missing_route','empty_prefix'):
        with pytest.raises(ValueError): m.accept(cases,prefixes)
    else: assert m.accept(cases,prefixes) == (fault=='none')


@pytest.mark.parametrize('fault', ['unbound','oracle','geometry'])
def test_reference_must_bind_current_oracle_and_short_geometry(fault):
    m=load()
    keys=('hidden','vocab','real_vocab','model','spec_hash','max_ctx')
    meta={k:k for k in keys}; short=dict(meta)
    meta['sources']={'oracle':'original'}
    proof=dict(passed=True, evidence={'/refs/255/reference.json':'ref'})
    if fault=='unbound': proof['evidence']['/refs/255/reference.json']='wrong'
    elif fault=='oracle': meta['sources']['oracle']='changed'
    else: meta['hidden']='wrong'
    with pytest.raises(ValueError):
        m.check_reference_binding(meta,short,proof,Path('/refs/255/reference.json'),'ref',{'oracle':'original'})


def test_completed_numerical_failure_can_be_audited_without_becoming_success(tmp_path,monkeypatch):
    m=load();cases=passing(m);cases['257']['routes'][0]['first']['passed']=False
    record=dict(complete=True,passed=False,model_acceptance=False,sources=m.sources(),
                prepared='/prepared',preparation_sha256='hash',cases=cases,prefixes={'prefix':True})
    (tmp_path/'report.json').write_text(json.dumps(record))
    monkeypatch.setattr(m,'verify_preparation',lambda p:({}, {}, None))
    monkeypatch.setattr(m,'measure',lambda *a:(cases,{'prefix':True}))
    original_digest=m.digest
    monkeypatch.setattr(m,'digest',lambda p:'hash' if str(p)=='/prepared/preparation.json' else original_digest(p))
    assert m.audit(SimpleNamespace(out=tmp_path))==1
    proof=json.loads((tmp_path/'audit.json').read_text())
    assert proof['complete'] and not proof['passed'] and not proof['model_acceptance']
    record['passed']=True
    (tmp_path/'report.json').write_text(json.dumps(record))
    with pytest.raises(ValueError): m.audit(SimpleNamespace(out=tmp_path))
    assert not json.loads((tmp_path/'audit.json').read_text())['complete']


def test_prefix_identity_excludes_different_continuations_but_checks_every_prompt_row(tmp_path,monkeypatch):
    m=load();monkeypatch.setattr(m,'LENGTHS',(2,3,4))
    cases={n:dict(hidden=2,layers=1) for n in m.LENGTHS}
    for n in cases:
        target=tmp_path/str(n)/'trace';target.mkdir(parents=True)
        for route in m.post.old.runner.ROUTES:
            for p in range(n):(target/f'{route}_p{p}.bin').write_bytes(bytes([p]))
            values=np.arange((n+1)*2,dtype='<f4');values[-2:]=100+n
            (target/f'{route}_res0.bin').write_bytes(values.tobytes())
    assert all(m.native_prefixes(tmp_path,cases).values())
    path=tmp_path/'3/trace/hybrid_res0.bin'
    values=np.fromfile(path,dtype='<f4');values[1]+=1;values.tofile(path)
    assert not all(m.native_prefixes(tmp_path,cases).values())
