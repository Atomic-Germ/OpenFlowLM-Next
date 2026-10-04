"""Residual substitutions separate propagated inputs from local projection errors."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from ml_dtypes import bfloat16


def module():
    path=Path(__file__).resolve().parents[3]/'utilities/diagnose-wide-residual.py'
    spec=importlib.util.spec_from_file_location('residual_boundary',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_counterfactuals_preserve_fp32_residual_rounding():
    m=module()
    x=np.array([1.,.1,-.2,3.],np.float32);p=np.array([.2,.4,.3,1.],np.float32)
    dx=x.copy();dx[0]=1.1;dp=p.copy();dp[2]=.4;w=np.ones(4,bfloat16)
    got=m.norm((dx+dp).astype(np.float32),w)
    r=m.analyze(dx,dp,x,p,w,got,0)
    assert r['diagnostic_only'] and not r['model_passed']
    assert r['counterfactuals']['device']['vs_device']==0
    assert r['counterfactuals']['reference']['vs_reference']==0
    assert r['counterfactuals']['input_only']['vs_reference']>0
    assert r['counterfactuals']['projection_only']['vs_reference']>0
    assert r['counterfactuals']['input_channel_only']['value']==r['counterfactuals']['input_only']['value']
    assert r['local_norm']==0
    np.testing.assert_array_equal(dx,np.array([1.1,.1,-.2,3.],np.float32))


def test_scalar_error_breakdown_distinguishes_rounded_sum():
    m=module()
    r=m.addition_error(np.float32(1),np.float32(2**-24),np.float32(1),np.float32(1),np.float32(0))
    assert r['local_addition']==0
    assert r['left_error']==0 and r['right_error']==2**-24
    assert r['rounding_delta']==-2**-24 and r['output_error']==0


def test_capture_rejects_guard_and_nonfinite(tmp_path):
    m=module();p=tmp_path/'capture';a=np.ones(4,np.float32)
    p.write_bytes(a.tobytes()+m.GUARD)
    np.testing.assert_array_equal(m.capture(p,4),a)
    p.write_bytes(a.tobytes()+bytes(64))
    with pytest.raises(ValueError,match='canary'):m.capture(p,4)
    a[0]=np.nan;p.write_bytes(a.tobytes()+m.GUARD)
    with pytest.raises(ValueError,match='nonfinite'):m.capture(p,4)


def test_down_rne_requires_segment_carry_before_build(tmp_path):
    import subprocess,sys
    root=Path(__file__).resolve().parents[3]
    p=subprocess.run([sys.executable,str(root/'utilities/probe-qwen35-wide.py'),
                      '--scope','ffn','--down-rne','--out',str(tmp_path/'absent')],capture_output=True,text=True)
    assert p.returncode==2 and '--down-rne requires --segment-carry' in p.stderr
    assert not (tmp_path/'absent').exists()


def test_real_fixture_padding_and_reference_tampering(tmp_path):
    import json
    m=module();tag='cold-0-layer0';layer=tmp_path/'layer0';layer.mkdir()
    x=np.ones(5120,np.float32);p=x/8;res=x+p;w=np.ones(5120,bfloat16);xm=m.norm(res,w)
    np.savez(layer/'params.npz',postw=w.astype(np.float32))
    np.savez(tmp_path/f'{tag}-ref.npz',x=x,projout=p,res1=res,fo=p,y=res+p,xm=xm.astype(np.float32))
    for field,a in [('input-x',x),('got-projout',p),('got-res1',res),('got-fo',p),('got-y',res+p)]:
        (tmp_path/f'{tag}-{field}.bin').write_bytes(a.tobytes()+m.GUARD)
    padded=np.r_[xm,np.full(1024,np.nan,bfloat16)].astype(bfloat16)
    (tmp_path/f'{tag}-got-xm.bin').write_bytes(padded.tobytes()+m.GUARD)
    names=['layer0/params.npz',f'{tag}-ref.npz']
    meta=dict(cases=[dict(tag=tag,layer=0,token='cold-0',kind=m.LINEAR)],kernels={},
              outputs={'d':{'xm':6144*2}},fixtures={n:m.sha(tmp_path/n) for n in names})
    (tmp_path/'slice-fixture.json').write_text(json.dumps(meta))
    r=m.diagnose(tmp_path,tag,4872)
    assert r['local_norm']==0 and all(v['output_error']==0 for v in r['channel_history'])
    (tmp_path/f'{tag}-ref.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='fixture changed'):m.diagnose(tmp_path,tag,4872)

