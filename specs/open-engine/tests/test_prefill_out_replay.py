"""Output replay distinguishes historical inputs, fresh builds and residual acceptance."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
def load(name='check'):
 s=importlib.util.spec_from_file_location('out_replay_'+name,ROOT/'utilities/prefill-out-replay'/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_output_geometry_and_clean_environment(monkeypatch):
 m=load('build');monkeypatch.setenv('GQP_K','5120');monkeypatch.setenv('GQP_BAD','1')
 e=m.environment();assert (e['GQP_N'],e['GQP_K'],e['GQP_T'],e['GQP_BREUSE'])==('5120','6144','256','0') and 'GQP_BAD' not in e

def test_build_protects_existing_evidence(tmp_path):
 with pytest.raises(FileExistsError):load('build').build(SimpleNamespace(out=tmp_path,mode='low'))

def test_local_projection_pass_does_not_hide_residual_failure():
 m=load();ref=np.ones((20,5120));x=np.ones(5120,np.float32)
 r=m.scores(ref*1.001,ref,ref,x,np.full(5120,2.02),17)
 assert r['projection_passed'] and r['improved'] and not r['residual']['passed']

def test_equal_candidate_is_not_improvement():
 m=load();ref=np.ones((20,5120));x=np.ones(5120,np.float32)
 assert not m.scores(ref,ref,ref,x,x*2,7)['improved']

@pytest.mark.parametrize('bad',['shape','nan','position'])
def test_scores_require_complete_finite_rows(bad):
 m=load();ref=np.ones((20,5120));got=ref.copy();p=17
 if bad=='shape':got=got[:-1]
 if bad=='nan':got[0,0]=np.nan
 if bad=='position':p=20
 with pytest.raises(ValueError):m.scores(got,ref,ref,np.ones(5120),np.ones(5120),p)

def test_historical_source_requires_exact_old_and_current_git_blobs(tmp_path,monkeypatch):
 m=load();monkeypatch.setattr(m,'ROOT',tmp_path);p=tmp_path/'source.py';p.write_bytes(b'new')
 monkeypatch.setattr(m,'git_bytes',lambda rev,name:b'old' if rev==m.FIXTURE_REV else b'new')
 result=m.verify_historical(p,hashlib.sha256(b'old').hexdigest())
 assert result['historical'] and result['current_sha256']==hashlib.sha256(b'new').hexdigest()
 p.write_bytes(b'dirty')
 with pytest.raises(ValueError):m.verify_historical(p,hashlib.sha256(b'old').hexdigest())

def test_wrong_historical_hash_rejected(tmp_path,monkeypatch):
 m=load();monkeypatch.setattr(m,'ROOT',tmp_path);p=tmp_path/'source.py';p.write_bytes(b'new')
 monkeypatch.setattr(m,'git_bytes',lambda rev,name:b'old')
 with pytest.raises(ValueError):m.verify_historical(p,'0'*64)

def test_artifact_corruption_never_uses_git_fallback(tmp_path,monkeypatch):
 m=load();monkeypatch.setattr(m,'ROOT',tmp_path);p=tmp_path/'Models'/'capture.bin';p.parent.mkdir();p.write_bytes(b'bad')
 monkeypatch.setattr(m,'git_bytes',lambda *args:pytest.fail('artifact must not use historical source fallback'))
 with pytest.raises(ValueError):m.verify_historical(p,'0'*64)

def test_fresh_build_rejects_wrong_shape(tmp_path,monkeypatch):
 m=load();monkeypatch.setattr(m.paired,'verify_build',lambda *a:dict(shape=dict(n=16384,k=5120,t=256),environment={}))
 with pytest.raises(ValueError):m.verify_build(tmp_path,'control')

def test_prepare_failure_persists(tmp_path):
 m=load();p=tmp_path/'prepared'
 with pytest.raises(FileNotFoundError):m.prepare(SimpleNamespace(out=p,baseline=tmp_path/'missing'))
 assert not json.loads((p/'preparation.json').read_text())['complete']
 with pytest.raises(FileExistsError):m.prepare(SimpleNamespace(out=p))

def test_run_failure_never_leaves_success(tmp_path):
 m=load();p=tmp_path/'run'
 with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=p,prepared=tmp_path/'missing'))
 r=json.loads((p/'report.json').read_text());assert not r['complete'] and not r['control_valid'] and not r['model_acceptance']

def test_history_graph_follows_pinned_records_without_guessing_run_binary(tmp_path):
 m=load();artifact=tmp_path/'output.bin';artifact.write_bytes(b'kept')
 child=tmp_path/'report.json';child.write_text(json.dumps(dict(cli_sha256='a'*64,artifacts={str(artifact):m.digest(artifact)})))
 root=tmp_path/'root.json';root.write_text(json.dumps(dict(evidence={str(child):m.digest(child),'sublayer_audit_sha256':'b'*64})))
 graph=m.history_graph(root);assert str(artifact) in graph['files']
 artifact.write_bytes(b'changed')
 with pytest.raises(ValueError):m.history_graph(root)
