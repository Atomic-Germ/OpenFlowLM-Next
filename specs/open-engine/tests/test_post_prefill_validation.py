"""New arithmetic requires fresh native identity and unchanged numerical gates."""
import importlib.util
from pathlib import Path
import pytest
import numpy as np
from types import SimpleNamespace
import json

ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('post_prefill',ROOT/'utilities/prefill-production/post_validate.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def test_plain_identity_covers_logits_and_final_scratch():
    m=load();names=m.old.runner.capture_names('sequential',2,3,False)
    plain={n:str(i) for i,n in enumerate(names)}
    traced=plain|{f'sequential_res{l}.bin':'residual' for l in range(3)}
    assert all(m.plain_identity(traced,plain,'sequential',2,3).values())
    for name in names:
        bad=dict(traced);bad[name]='changed'
        with pytest.raises(ValueError):m.plain_identity(bad,plain,'sequential',2,3)
    for bad in ({**traced,'sequential_res3.bin':'extra'}, {k:v for k,v in traced.items() if k!='sequential_res1.bin'}):
        with pytest.raises(ValueError):m.plain_identity(bad,plain,'sequential',2,3)


@pytest.mark.parametrize('fault',['plain','trace','identity'])
def test_numeric_or_identity_failure_cannot_be_hidden(fault):
    m=load();plain=[dict(route=r,passed=True) for r in m.old.runner.ROUTES]
    traced=[dict(route=r,first={'passed':True},identity={'logits':True}) for r in m.old.runner.ROUTES]
    assert m.accept(plain,traced)
    if fault=='plain':plain[0]['passed']=False
    elif fault=='trace':traced[0]['first']['passed']=False
    else:traced[0]['identity']['logits']=False
    with pytest.raises(ValueError):m.accept(plain,traced)


def test_missing_plain_route_is_rejected():
    m=load()
    with pytest.raises(ValueError):m.accept([],[])


def test_plain_inventory_rejects_extra_residuals(tmp_path):
    m=load()
    for n in m.old.runner.capture_names('hybrid',2,3,False):(tmp_path/n).touch()
    m.inventory(tmp_path,'hybrid',2,3,False)
    (tmp_path/'hybrid_res0.bin').touch()
    with pytest.raises(ValueError):m.inventory(tmp_path,'hybrid',2,3,False)


@pytest.mark.parametrize('fault',['logit','residual','alias','nan'])
def test_plain_numerical_gate_and_alias_are_required(tmp_path,fault):
    m=load();out=tmp_path/'out';refs=tmp_path/'refs';out.mkdir();refs.mkdir()
    values=np.array([0,1,2],dtype='<f4')
    for name in m.base.expected_dumps('sequential',2):(out/name).write_bytes(values.tobytes())
    for name in ('ref_p0.bin','ref_p1.bin','ref_cont_logits.bin'):(refs/name).write_bytes(values.tobytes())
    (refs/'ref_cont_res2.bin').write_bytes(np.array([1],dtype='<f4').tobytes())
    (out/'sequential-act.bin').write_bytes(np.array([1,0,0],dtype='<f4').tobytes())
    (out/'sequential.log').write_text('token 2\ntoken 2\n')
    meta=dict(ids=[1,2],layers=3,hidden=1,vocab=3,real_vocab=3,expected_tokens=[2,2])
    offsets=SimpleNamespace(AA_BYTES=12,AA_RES=0,AA_OUT2=4,AA_OUT2B=8)
    assert m.plain_score(out,refs,meta,offsets,'sequential')['passed']
    if fault=='residual':(out/'sequential-act.bin').write_bytes(np.array([1.01,0,0],dtype='<f4').tobytes())
    else:
        data=np.array([0,1.2,2] if fault!='nan' else [0,1,np.nan],dtype='<f4')
        name='sequential_t0.bin' if fault=='alias' else 'sequential_p0.bin'
        (out/name).write_bytes(data.tobytes())
    if fault=='nan':
        with pytest.raises(ValueError):m.plain_score(out,refs,meta,offsets,'sequential')
    else:assert not m.plain_score(out,refs,meta,offsets,'sequential')['passed']


def test_audit_context_rejects_replaced_reference_inventory(monkeypatch):
    m=load();monkeypatch.setattr(m.base,'verify',lambda _:None)
    monkeypatch.setattr(m.builder,'verify',lambda *a,**kw:{'cli_sha256':'cli'})
    monkeypatch.setattr(m,'fixture',lambda *a:({},None,None,{'weights':'new'}))
    record=dict(sources={},plain_build='plain',trace_build='trace',plain_sha256='cli',trace_sha256='cli',
                baseline='baseline',kernels='kernels',inputs={'weights':'old'})
    with pytest.raises(ValueError,match='inputs changed'):m.context(record)


def test_failed_reaudit_replaces_prior_success(tmp_path):
    m=load()
    (tmp_path/'report.json').write_text(json.dumps(dict(complete=False)))
    (tmp_path/'audit.json').write_text(json.dumps(dict(complete=True,passed=True)))
    with pytest.raises(ValueError):m.audit(SimpleNamespace(out=tmp_path))
    proof=json.loads((tmp_path/'audit.json').read_text())
    assert not proof['passed'] and not proof['complete'] and not proof['model_acceptance']
