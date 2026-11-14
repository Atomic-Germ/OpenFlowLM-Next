"""Projection captures must fail separately even when their product cancels."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

_spec=importlib.util.spec_from_file_location('front_probe_check',Path(__file__).resolve().parents[3]/'utilities/ffn-front-replay/check.py')
mod=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(mod)


def test_reads_all_three_interleaved_band_outputs(tmp_path):
    a=np.arange(384,dtype=np.float32).reshape(2,3,64)
    p=tmp_path/'out';p.write_bytes(a.tobytes())
    u,g,h=mod.read_outputs(p,128)
    for i,got in enumerate((u,g,h)):np.testing.assert_array_equal(got,a[:,i,:].reshape(-1))
    p.write_bytes(a.tobytes()[:-4])
    with pytest.raises(ValueError):mod.read_outputs(p,128)


def test_product_cannot_hide_wrong_up_projection():
    res=np.ones(2);nw=np.ones(2);x=res/np.sqrt(1+1e-6)
    up=np.ones((1,2));gate=up.copy();u=up@x;g=gate@x;h=u*g/(1+np.exp(-g))
    r=mod.score_arrays(res,nw,up,gate,x,(u*2,g,h),1e-6)
    assert not r['passed']
    assert r['front']['passed']
    assert not r['up']['passed']


def test_exact_front_and_projections_pass():
    res=np.ones(2);nw=np.ones(2);x=res/np.sqrt(1+1e-6)
    w=np.ones((1,2));u=w@x;h=u*u/(1+np.exp(-u))
    assert mod.score_arrays(res,nw,w,w,x,(u,u,h),1e-6)['passed']


def test_input_digest_change_refused(tmp_path):
    import hashlib,json
    p=tmp_path/'input';p.write_bytes(b'original')
    audit={'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()}}
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    mod.verify(tmp_path)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):mod.verify(tmp_path)


def test_missing_input_coverage_refused(tmp_path):
    import json
    (tmp_path/'audit.json').write_text(json.dumps({'sha256':{}}))
    with pytest.raises(ValueError,match='coverage'):
        mod.verify(tmp_path,[tmp_path/'input'])


@pytest.mark.parametrize('kind',[0,1,2])
def test_nonfinite_output_refused(tmp_path,kind):
    a=np.zeros((1,3,64),np.float32);a[0,kind,0]=np.nan
    p=tmp_path/'out';p.write_bytes(a.tobytes())
    with pytest.raises(ValueError,match='finite'):mod.read_outputs(p,64)


def test_bad_fixture_invalidates_previous_success(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();(out/'score.json').write_text('{"passed":true}')
    args=['check','--model-dir',str(tmp_path/'model'),'--kernel-dir',str(tmp_path/'kernels'),
          '--fixture',str(tmp_path/'fixture'),'--pools',str(tmp_path/'pools'),
          '--out',str(out),'--build',str(tmp_path/'build')]
    monkeypatch.setattr('sys.argv',args)
    def invalid(_):raise ValueError('bad fixture')
    monkeypatch.setattr(mod,'spec_from_model_dir',invalid)
    with pytest.raises(ValueError):mod.main()
    assert not (out/'score.json').exists()
