"""Selected drift replay must handle layer zero and pin exact target coverage."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('drift_targets', ROOT/'utilities/prefill-sequential-drift/targets.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def test_targets_are_normalized_and_cover_actual_prompt_only():
    m = load()
    assert m.parse_targets(['59:39,37,38','0:241','32:240,241']) == {0:(241,),32:(240,241),59:(37,38,39)}


@pytest.mark.parametrize('targets', [[], ['-1:1'], ['64:1'], ['0:-1'], ['0:257'], ['0:'], ['0:1,1'], ['0:1','0:2'], ['0:1.5']])
def test_invalid_or_ambiguous_targets_are_refused(targets):
    with pytest.raises(ValueError): load().parse_targets(targets)


def test_layer_zero_uses_shared_fp32_embeddings_without_residual_files(tmp_path, monkeypatch):
    m = load(); seen = []
    model = SimpleNamespace(embed=lambda t,h: seen.append(t) or np.array([t+.1,t+.2],np.float64))
    meta = dict(ids=[3,1,4], hidden=2)
    monkeypatch.setattr(m.drift.trace, 'read_trace', lambda *a: (_ for _ in ()).throw(AssertionError('read layer -1')))
    values = [m.prefix_inputs(0,r,model,meta,tmp_path,tmp_path) for r in ('reference',*m.drift.ROUTES)]
    assert all(x.dtype == np.float32 and x.shape == (3,2) for x in values)
    assert all(x.tobytes() == values[0].tobytes() for x in values)
    values[0][0,0] = -1
    assert values[1][0,0] > 0 and seen == [3,1,4]*3


def test_nonzero_layer_uses_preceding_causal_residual_and_drops_continuation(tmp_path):
    m = load(); meta=dict(ids=[1,2,3],hidden=2)
    ref=np.arange(6,dtype='<f4').reshape(3,2)
    ref.tofile(tmp_path/'ref_res31.bin')
    np.vstack((ref+1,[999,999])).astype('<f4').tofile(tmp_path/'sequential_res31.bin')
    np.testing.assert_array_equal(m.prefix_inputs(32,'reference',None,meta,tmp_path,tmp_path),ref)
    np.testing.assert_array_equal(m.prefix_inputs(32,'sequential',None,meta,tmp_path,tmp_path),ref+1)


def test_layer_zero_refuses_nonzero_propagated_components():
    m=load(); vectors={k:np.zeros(2) for k in ('current_input','input_history','propagated')}
    m.check_embedding_identity(0,vectors)
    vectors['input_history'][0]=1
    with pytest.raises(ValueError): m.check_embedding_identity(0,vectors)
    m.check_embedding_identity(32,vectors)


def test_summary_cannot_skip_selected_case():
    m=load(); targets={0:(1,),32:(2,3)}
    rows=[dict(layer=l,position=p,route=r,metrics={k:dict(maxrel=0.,passed=True) for k in (*m.drift.PARTS,'propagated','total')}) for l,ps in targets.items() for p in ps for r in m.drift.ROUTES]
    assert m.summarize(rows,targets)['sequential']['total']==3
    with pytest.raises(ValueError):m.summarize(rows[:-1],targets)
    with pytest.raises(ValueError):m.summarize(rows+[rows[0]],targets)


def test_failure_is_persisted_and_cannot_overwrite(tmp_path):
    m=load();a=SimpleNamespace(out=tmp_path/'run',baseline=tmp_path/'missing',target=['0:241'])
    with pytest.raises(FileNotFoundError):m.run(a)
    r=json.loads((a.out/'report.json').read_text());assert not r['complete'] and not r['model_acceptance']
    with pytest.raises(FileExistsError):m.run(a)
