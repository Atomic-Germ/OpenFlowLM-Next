"""Unrounded entry references must expose norm-induced projection errors."""
from types import SimpleNamespace
import numpy as np
import pytest
from attention_projections import compare_entry, compare_output, read_boundaries
from q4nx import f32_to_bf16, bf16_to_f32


def entry():
    x = np.array([1., -2.]); norm = np.array([.7, 1.1]); eps = 1e-6
    xn = x / np.sqrt(np.mean(x*x)+eps)*norm
    w = {'q': np.array([[1., 2.], [-3., 1.]]), 'gate': np.eye(2),
         'k': np.array([[2., 1.]]), 'v': np.array([[-1., 2.]])}
    return [x,norm,w,xn,{k:v@xn for k,v in w.items()},eps]


def test_exact_independent_entry():
    r=compare_entry(*entry())
    assert r['passed'] and set(r['metrics'])=={'norm','q','gate','k','v'}
    assert max(m['maxrel'] for m in r['metrics'].values())<1e-14


def test_cancellation_after_bf16_norm_cannot_hide_behind_conditioned_matmul():
    x=np.ones(2);norm=np.array([1.,1.001]);eps=1e-6
    xn=bf16_to_f32(f32_to_bf16((x/np.sqrt(1+eps)*norm).astype(np.float32)))
    w={k:np.array([[1.,-1.]]) for k in ('q','gate','k','v')}
    r=compare_entry(x,norm,w,xn,{k:v@xn for k,v in w.items()},eps)
    assert not r['passed'] and not r['metrics']['q']['passed']
    assert r['conditioned']['q']['passed'] and r['rounded_norm_diagnostic']['q']['passed']


@pytest.mark.parametrize('key',['q','gate','k','v'])
def test_each_projection_is_required(key):
    a=entry();a[4][key]+=1
    assert not compare_entry(*a)['passed']
    a=entry();del a[4][key]
    with pytest.raises(ValueError):compare_entry(*a)


@pytest.mark.parametrize('index',[0,1,3])
@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_invalid_entry_values(index,value):
    a=entry();a[index][0]=value
    with pytest.raises(ValueError,match='finite'):compare_entry(*a)


@pytest.mark.parametrize('eps',[0,-1,np.nan,np.inf])
def test_invalid_epsilon(eps):
    a=entry();a[-1]=eps
    with pytest.raises(ValueError,match='epsilon'):compare_entry(*a)


def test_entry_shapes_and_nonfinite_weight():
    a=entry();a[2]['q']=np.ones((3,2))
    with pytest.raises(ValueError,match='shape'):compare_entry(*a)
    a=entry();a[2]['q'][0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):compare_entry(*a)


def test_output_projection_and_residual_have_separate_gates():
    x=np.array([1.,2.]);og=np.array([.3,.7]);w=np.array([[1.,2.],[3.,4.]])
    out=w@og;res=x+out
    assert compare_output(x,og,w,out,res)['passed']
    assert not compare_output(x,og,w,out+1,res)['passed']
    assert not compare_output(x,og,w,out,res+1)['passed']
    # Input rounding would lose this cancellation, so compare to the FP64 OG.
    og=np.array([1.,1.001]);w=np.array([[1.,-1.],[1.,-1.]])
    assert not compare_output(x,og,w,np.zeros(2),x)['passed']


def test_zero_reference_requires_exact_zero_and_finite():
    z=np.zeros(2);w=np.eye(2)
    assert compare_output(z,z,w,z,z)['passed']
    assert not compare_output(z,z,w,z+1e-30,z)['passed']
    with pytest.raises(ValueError,match='finite'):compare_output(z,z,w,z+np.nan,z)
    with pytest.raises(ValueError,match='shape'):compare_output(z,z,w,z[:1],z)


def test_read_boundaries_and_reject_bad_capture(tmp_path):
    o=SimpleNamespace(AA_BYTES=24,AA_XN=0,AA_OUT=8,AA_RES=16)
    b=bytearray(24);b[:4]=f32_to_bf16(np.array([1.,2.],np.float32)).tobytes()
    b[8:]=np.array([3.,4.,5.,6.],np.float32).tobytes();p=tmp_path/'act';p.write_bytes(b)
    r=read_boundaries(p,o,2);np.testing.assert_array_equal(r['xn'],[1,2]);np.testing.assert_array_equal(r['res'],[5,6])
    p.write_bytes(b[:-1])
    with pytest.raises(ValueError,match='bytes'):read_boundaries(p,o,2)
    p.write_bytes(b);o.AA_OUT=20
    with pytest.raises(ValueError,match='region'):read_boundaries(p,o,2)


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    import json
    import attention_projections as mod
    from recipes.spec import FULL, LINEAR
    spec=SimpleNamespace(family='qwen35',hidden=2,num_layers=4,layer_types=[LINEAR,FULL,LINEAR,FULL],
                         attn_q_width=2,attn_kv_width=2,num_heads=1,num_kv_heads=1,head_dim=2,norm_eps=1e-6,rotary_dim=0,rope_inv_freq=lambda:np.zeros(0))
    meta=dict(hidden=2,layers=4,tokens=2,max_ctx=4,spec_hash='s',build_key='b')
    offsets=SimpleNamespace(AA_BYTES=64,AA_XN=0,AA_QG=8,AA_KVN=24,AA_OG=40,AA_OUT=48,AA_RES=56)
    (tmp_path/'decode_reference.json').write_text(json.dumps(meta))
    for l in [1,3]:
        for t in range(2):
            suffix='' if not t else '_t1'
            (tmp_path/f'y_res{l-1}{suffix}.bin').write_bytes(np.ones(2,np.float32).tobytes())
            b=bytearray(64);b[:4]=f32_to_bf16(np.ones(2,np.float32)).tobytes()
            for off,v in [(8,[1,1,2,2]),(24,[1,1,3,3]),(40,[.3,.7]),(48,[.6,1.4]),(56,[1.6,2.4])]:
                b[off:off+len(v)*4]=np.array(v,np.float32).tobytes()
            og=np.full(2,3/(1+np.exp(-2)),np.float32)
            b[40:48]=og.tobytes();b[48:56]=(2*og).tobytes();b[56:64]=(1+2*og).tobytes()
            (tmp_path/f'y_act{l}{suffix}.bin').write_bytes(b)
    class Model:
        def __init__(self,*args):pass
        def bf16(self,name):return np.ones(2,np.float32)
        def matmul_w(self,name,*args):
            if 'q_proj' in name:return np.vstack([np.eye(2),2*np.eye(2)])
            return (3 if 'v_proj' in name else 2 if 'o_proj' in name else 1)*np.eye(2)
    monkeypatch.setattr(mod,'spec_from_model_dir',lambda *a:spec)
    monkeypatch.setattr(mod,'norm_split',lambda *a:True)
    monkeypatch.setattr(mod,'load_reference',lambda *a:meta)
    monkeypatch.setattr(mod,'manifest',lambda *a,**k:meta)
    monkeypatch.setattr(mod,'load_bundle',lambda *a:dict(meta,layout={'attn_output_dtype':'float32'}))
    monkeypatch.setattr(mod,'layout',lambda *a,**k:offsets)
    monkeypatch.setattr(mod,'Q4NX',Model)
    return mod,tmp_path,meta


def test_complete_coverage_provenance_and_missing_capture(bundle):
    mod,p,_=bundle;r=mod.check(p,p,p)
    assert r['passed'] and len(r['results'])==4 and r['layers']==[1,3]
    assert len(r['capture_sha256'])==8 and len(r['weight_sha256'])==14
    (p/'y_act3_t1.bin').unlink()
    with pytest.raises(OSError):mod.check(p,p,p)


@pytest.mark.parametrize('layers',[[],[0],[4],[1,1]])
def test_invalid_coverage(bundle,layers):
    mod,p,_=bundle
    with pytest.raises(ValueError,match='indices'):mod.check(p,p,p,layers)


@pytest.mark.parametrize('key',['build_key','spec_hash'])
def test_stale_export(bundle,monkeypatch,key):
    mod,p,m=bundle
    monkeypatch.setattr(mod,'load_bundle',lambda *a:dict(m,**{key:'stale'}))
    with pytest.raises(ValueError,match=key):mod.check(p,p,p)


def test_reject_overwritten_entry_norm(bundle,monkeypatch):
    mod,p,_=bundle;monkeypatch.setattr(mod,'norm_split',lambda *a:False)
    with pytest.raises(ValueError,match='retained'):mod.check(p,p,p)


def test_cli_primary_failure_returns_one(bundle,monkeypatch,capsys):
    import json
    mod,p,_=bundle;f=p/'y_act3_t1.bin';b=bytearray(f.read_bytes());b[8:12]=np.array([100.],np.float32).tobytes();f.write_bytes(b)
    monkeypatch.setattr('sys.argv',['checker','--model-dir',str(p),'--kernel-dir',str(p),'--out',str(p)])
    assert mod.main()==1
    result=json.loads(capsys.readouterr().out)
    assert not result['passed'] and not result['results'][-1]['entry']['metrics']['q']['passed']


def test_sublayer_catches_error_hidden_by_conditioned_output_projection(bundle):
    mod,p,_=bundle
    f=p/'y_act3_t1.bin';b=bytearray(f.read_bytes())
    og=np.array([1.,2.],np.float32)
    b[40:48]=og.tobytes();b[48:56]=(2*og).tobytes();b[56:64]=(1+2*og).tobytes();f.write_bytes(b)
    r=mod.check(p,p,p)['results'][-1]
    assert r['entry']['passed'] and r['output']['passed']
    assert not r['sublayer']['passed'] and not r['passed']


def test_sublayer_recomputes_kv_history_from_incoming_residuals(bundle):
    mod,p,_=bundle
    (p/'y_res2.bin').write_bytes(np.array([2.,1.],np.float32).tobytes())
    row=mod.check(p,p,p)['results'][-1]
    assert row['entry']['passed'] and row['output']['passed']
    assert not row['sublayer']['passed']


def test_nonfinite_capture_refused(bundle):
    mod,p,_=bundle;f=p/'y_act3_t1.bin';b=bytearray(f.read_bytes());b[48:52]=np.array([np.nan],np.float32).tobytes();f.write_bytes(b)
    with pytest.raises(ValueError,match='finite'):mod.check(p,p,p)


def test_dimension_mismatch_refused(bundle):
    mod,p,m=bundle;m['hidden']=3
    with pytest.raises(ValueError,match='dimensions'):mod.check(p,p,p)


def test_checker_uses_declared_fp32_entry(bundle, monkeypatch):
    mod, p, meta = bundle
    monkeypatch.setattr(mod, 'load_bundle', lambda *args: dict(
        meta, layout={'attn_output_dtype': 'float32', 'attn_entry_dtype': 'float32'}))
    for capture in p.glob('y_act*.bin'):
        data = bytearray(capture.read_bytes())
        data[:8] = np.ones(2, np.float32).tobytes()
        capture.write_bytes(data)
    result = mod.check(p, p, p)
    assert result['passed']
    assert all(row['entry']['metrics']['norm']['maxrel'] < 1e-5 for row in result['results'])
