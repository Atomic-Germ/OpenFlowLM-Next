"""Per-head gates must catch errors hidden by aggregate attention/logit scores."""
from types import SimpleNamespace
import json
import numpy as np
import pytest

from attention_heads import attention_reference, compare_heads, read_attention
from q4nx import f32_to_bf16


def inputs():
    rng = np.random.default_rng(73)
    return [rng.normal(size=(3, 4, 4)), rng.normal(size=(3, 2, 4)),
            rng.normal(size=(3, 2, 4)), rng.normal(size=(3, 4, 4)),
            np.array([.7, 1.1, .8, 1.3]), np.array([1.2, .9, .6, 1.4]),
            1e-6, 2, np.array([.7])]


def scalar_oracle(args):
    q,k,v,g,qn,kn,eps,rot,inv = args
    def transform(a, w, t):
        a = a / np.sqrt(sum(float(x)**2 for x in a) / len(a) + eps) * w
        b = a.copy()
        for i in range(rot//2):
            c,s = np.cos(t*inv[i]), np.sin(t*inv[i])
            b[i] = a[i]*c-a[i+rot//2]*s
            b[i+rot//2] = a[i]*s+a[i+rot//2]*c
        return b
    out=np.empty_like(q)
    for t in range(len(q)):
        for h in range(q.shape[1]):
            kh=h//(q.shape[1]//k.shape[1]); qq=transform(q[t,h],qn,t)
            scores=np.array([sum(qq*transform(k[j,kh],kn,j))/np.sqrt(q.shape[2]) for j in range(t+1)])
            p=np.exp(scores-scores.max());p/=p.sum()
            out[t,h]=sum(p[j]*v[j,kh] for j in range(t+1))/(1+np.exp(-g[t,h]))
    return out


def test_fp64_reference_matches_scalar_causal_gqa_rope():
    args=inputs();out=attention_reference(*args)
    np.testing.assert_allclose(out,scalar_oracle(args),rtol=1e-13,atol=1e-13)
    assert out.dtype==np.float64
    assert all(r['passed'] for r in compare_heads(out,out))


def test_first_token_groups_and_future_cannot_change_past():
    args=inputs();out=attention_reference(*args)
    expected=np.repeat(args[2][0],2,axis=0)/(1+np.exp(-args[3][0]))
    np.testing.assert_allclose(out[0],expected)
    for a in args[:4]:a[2]*=100
    np.testing.assert_array_equal(attention_reference(*args)[:2],out[:2])


def test_history_changes_later_position():
    args=inputs();out=attention_reference(*args);args[2][0]+=10
    assert not np.allclose(attention_reference(*args)[1],out[1])


def test_every_head_scored_and_permutation_fails():
    ref=attention_reference(*inputs());got=ref.copy();got[2,[0,3]]=got[2,[3,0]]
    rows=compare_heads(got,ref)
    assert len(rows)==12
    assert {(r['token'],r['head']) for r in rows if not r['passed']}=={(2,0),(2,3)}


def test_max_error_and_cosine_are_independent_strict_gates():
    ref=np.array([[[1.,0.]]])
    assert not compare_heads(ref*1.03,ref)[0]['passed']
    r=compare_heads(np.array([[[1.,.019]]]),ref)[0]
    assert r['maxabs']<.02 and r['cosine']<.9999 and not r['passed']


@pytest.mark.parametrize('delta,passed',[(.019,True),(.021,False)])
def test_absolute_bound(delta,passed):
    ref=np.array([[[100.,50.]]])
    assert compare_heads(ref+delta,ref)[0]['passed']==passed


def test_small_relative_error_cannot_hide_absolute_head_failure():
    ref=np.array([[[100.,50.]]]);got=ref+0.03
    r=compare_heads(got,ref)[0]
    assert r['maxrel']<.02 and r['cosine']>.9999
    assert not r['passed']


def test_bf16_output_diagnostic_cannot_override_primary_failure():
    ref=np.array([[[100.03,50.]]])
    row=compare_heads(np.array([[[100.,50.]]]),ref)[0]
    assert not row['passed']
    assert row['ideal_bf16_output_maxabs']>.02
    assert row['error_to_bf16_reference_maxabs']==0


def test_zero_reference_requires_exact_zero():
    z=np.zeros((1,1,2));assert compare_heads(z,z)[0]['passed']
    r=compare_heads(z+1e-30,z)[0]
    assert not r['passed'] and r['maxrel'] is None
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('index',range(6))
@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_nonfinite_reference_inputs_rejected(index,value):
    args=inputs();args[index].flat[0]=value
    with pytest.raises(ValueError,match='finite'):attention_reference(*args)


@pytest.mark.parametrize('eps',[0,-1,np.nan])
def test_invalid_epsilon(eps):
    args=inputs();args[6]=eps
    with pytest.raises(ValueError,match='epsilon'):attention_reference(*args)


@pytest.mark.parametrize('rot',[-2,1,6])
def test_invalid_rotation(rot):
    args=inputs();args[7]=rot
    with pytest.raises(ValueError,match='rotary'):attention_reference(*args)


def test_bad_shapes_and_gqa_ratio():
    args=inputs();args[0]=args[0][:,:3];args[3]=args[3][:,:3]
    with pytest.raises(ValueError,match='shape'):attention_reference(*args)
    with pytest.raises(ValueError,match='shape'):compare_heads(np.zeros((1,1,0)),np.zeros((1,1,0)))
    with pytest.raises(ValueError,match='shape'):compare_heads(np.ones((1,1,2)),np.ones((1,2,2)))
    with pytest.raises(ValueError,match='finite'):compare_heads(np.full((1,1,2),np.nan),np.ones((1,1,2)))


def test_extreme_gate_is_stable():
    args=inputs();args[3][:]=[-1e4,1e4,-1e4,1e4]
    with np.errstate(over='raise',invalid='raise'):
        out=attention_reference(*args)
    assert np.isfinite(out).all() and np.all(out[:,:,[0,2]]==0)


@pytest.fixture
def tiny(tmp_path,monkeypatch):
    import attention_heads as mod
    args=inputs();ref=attention_reference(*args)
    spec=SimpleNamespace(family='qwen35',num_heads=4,num_kv_heads=2,head_dim=4,
        hidden=8,num_layers=2,layer_types=['linear_attention','full_attention'],
        rotary_dim=2,norm_eps=1e-6,rope_inv_freq=lambda:args[8])
    offsets=SimpleNamespace(AA_QG=8,AA_KVN=136,AA_OG=200,AA_BYTES=240)
    meta=dict(layers=2,tokens=3,hidden=8,max_ctx=4,spec_hash='s',build_key='b')
    monkeypatch.setattr(mod,'spec_from_model_dir',lambda _:spec)
    monkeypatch.setattr(mod,'manifest',lambda *a,**k:meta.copy())
    monkeypatch.setattr(mod,'load_bundle',lambda *a:meta.copy())
    monkeypatch.setattr(mod,'load_reference',lambda _:meta.copy())
    monkeypatch.setattr(mod,'layout',lambda *a,**k:offsets)
    monkeypatch.setattr(mod,'Q4NX',lambda _:SimpleNamespace(bf16=lambda n:args[4] if '.q_norm.' in n else args[5]))
    (tmp_path/'decode_reference.json').write_text('{}')
    for t in range(3):
        data=bytearray(offsets.AA_BYTES)
        data[8:136]=np.concatenate([args[0][t],args[3][t]]).astype(np.float32).tobytes()
        data[136:200]=np.concatenate([args[1][t],args[2][t]]).astype(np.float32).tobytes()
        data[200:232]=f32_to_bf16(ref[t].astype(np.float32)).tobytes()
        (tmp_path/f'y_act1{mod.sfx(t)}.bin').write_bytes(data)
    return mod,tmp_path,spec,offsets,meta


def test_capture_layout_and_full_coverage(tiny):
    mod,p,spec,offsets,meta=tiny
    r=read_attention(p/'y_act1.bin',offsets,4,2,4)
    assert r['q'].shape==(4,4) and r['v'].shape==(2,4)
    result=mod.check(p,p,p)
    assert result['passed'] and len(result['results'])==12
    assert len(result['capture_sha256'])==3
    assert len(result['norm_sha256'])==2
    (p/'y_act1_t2.bin').unlink()
    with pytest.raises(OSError):mod.check(p,p,p)


@pytest.mark.parametrize('kind',['truncated','nan','bounds'])
def test_bad_capture(tiny,kind):
    mod,p,spec,o,meta=tiny;f=p/'y_act1.bin';data=bytearray(f.read_bytes())
    if kind=='truncated':data=data[:-1]
    elif kind=='nan':data[200:202]=np.array([0x7fc0],np.uint16).tobytes()
    else:o.AA_OG=230
    f.write_bytes(data)
    with pytest.raises(ValueError):read_attention(f,o,4,2,4)


@pytest.mark.parametrize('layers',[[],[0],[2],[1,1]])
def test_invalid_layer_selection(tiny,layers):
    mod,p,*_=tiny
    with pytest.raises(ValueError,match='layers'):mod.check(p,p,p,layers)


@pytest.mark.parametrize('key',['build_key','spec_hash'])
def test_stale_bundle(tiny,monkeypatch,key):
    mod,p,spec,o,meta=tiny
    monkeypatch.setattr(mod,'load_bundle',lambda *a:dict(meta,**{key:'changed'}))
    with pytest.raises(ValueError,match=key):mod.check(p,p,p)


def test_cli_reports_bad_head_nonzero(tiny,monkeypatch,capsys):
    mod,p,*_=tiny;f=p/'y_act1_t2.bin';data=bytearray(f.read_bytes())
    data[200:208]=f32_to_bf16(np.full(4,100,np.float32)).tobytes();f.write_bytes(data)
    monkeypatch.setattr('sys.argv',['attention_heads','--model-dir',str(p),'--kernel-dir',str(p),'--out',str(p)])
    assert mod.main()==1
    result=json.loads(capsys.readouterr().out)
    assert not result['passed'] and len(result['results'])==12
    assert not result['results'][8]['passed']
