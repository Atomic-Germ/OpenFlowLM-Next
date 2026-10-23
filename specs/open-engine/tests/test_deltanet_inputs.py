"""Unrounded record preparation and causal state must remain primary gates."""
from types import SimpleNamespace
import json
import numpy as np
import pytest

import deltanet_inputs as mod
from deltanet_state import step_reference
from q4nx import bf16_to_f32, f32_to_bf16
from recipes.spec import LINEAR, FULL


def spec():
    return SimpleNamespace(hidden=2, norm_eps=1e-6, lin_key_heads=2, lin_value_heads=6,
                           lin_key_dim=128, lin_value_dim=128, lin_qkv_dim=1280, conv_kernel=4)


def weights(s):
    rng = np.random.default_rng(42)
    return dict(norm=np.ones(s.hidden), qkv=rng.normal(0, .2, (s.lin_qkv_dim, s.hidden)),
                conv=np.ones((4, s.lin_qkv_dim))*.25, alpha=np.zeros((6, s.hidden)),
                beta=np.zeros((6, s.hidden)), a=-np.ones(6), dt=np.zeros(6))


def test_records_use_key_grouping_and_stable_ab():
    s = spec(); w = weights(s); w['conv'][:] = 0; w['conv'][-1] = 1
    qkv = np.zeros(s.lin_qkv_dim)
    qkv[[0, 129, 258, 387]] = 1  # two distinct Q and K heads
    qkv[512:] = 2
    r, history = mod.records_reference(s, np.ones(2), qkv, np.zeros((3, 1280)), w)
    assert r.shape == (6, 512) and r.dtype == np.float64
    assert np.argmax(r[0, :128]) == 2 and np.argmax(r[3, :128]) == 3
    assert np.argmax(r[0, 128:256]) == 0 and np.argmax(r[3, 128:256]) == 1
    np.testing.assert_array_equal(r[0, :256], r[2, :256])
    np.testing.assert_array_equal(r[3, :256], r[5, :256])
    np.testing.assert_allclose(r[:, 256:384], 2/(1+np.exp(-2)))
    np.testing.assert_array_equal(r[:, 384:386], .5)
    np.testing.assert_array_equal(history[-1], qkv)
    assert not r[:, 386:].any()
    w['alpha'][:] = 1000; w['beta'][:] = -1000
    r, _ = mod.records_reference(s, np.ones(2), qkv, history, w)
    assert np.isfinite(r).all() and not r[:, 384:386].any()


def test_conv_history_is_unrounded_and_causal():
    s = spec(); w = weights(s); w['conv'][:] = 0; w['conv'][-2] = 1
    q = np.full(1280, 1.001)
    r0, history = mod.records_reference(s, np.ones(2), q, np.zeros((3, 1280)), w)
    r1, _ = mod.records_reference(s, np.ones(2), np.zeros(1280), history, w)
    assert not r0[:, :384].any()
    np.testing.assert_allclose(r1[:, 256:384], 1.001/(1+np.exp(-1.001)))
    assert history[-1, 0] == 1.001


def captures(s, w, inputs):
    history = np.zeros((3, s.lin_qkv_dim)); state = np.zeros((6, 128, 128)); out = []
    for x in inputs:
        xn = x/np.sqrt(np.mean(x*x)+s.norm_eps)*w['norm']
        qkv = w['qkv']@xn
        r, history = mod.records_reference(s, xn, qkv, history, w)
        state, o = step_reference(state, r)
        out.append(dict(xn=xn, qkv=qkv, records=r, conv=f32_to_bf16(history.astype(np.float32)),
                        state=np.pad(state, ((0,0),(0,2),(0,0))), output=o))
    return out


def test_primary_sequence_matches_exact_unrounded_reference():
    s = spec(); w = weights(s); x = [np.array([1., -.5]), np.array([.25, 2.])]
    rows = mod.score_sequence(s, w, x, captures(s,w,x))
    assert all(r['primary_passed'] for r in rows)
    assert all(r['composed_state']['maxrel'] == 0 for r in rows)


def test_conditioned_pass_cannot_override_entry_rounding_failure():
    s = spec(); w = weights(s); w['norm'] = np.array([1.,1.001])
    w['qkv'][:] = [1.,-1.]; x = [np.ones(2),np.ones(2)]
    c = captures(s,w,x)
    for row in c:
        row['xn'] = bf16_to_f32(f32_to_bf16(row['xn'].astype(np.float32)))
        row['qkv'] = w['qkv']@row['xn']
    rows = mod.score_sequence(s,w,x,c)
    assert all(not r['primary_passed'] for r in rows)
    assert all(r['conditioned_qkv']['passed'] for r in rows)
    assert all(r['rounded_norm_qkv']['passed'] for r in rows)


def test_captured_previous_state_cannot_change_primary_reference():
    s = spec(); w = weights(s); x = [np.array([1., -.5])]*3
    c = captures(s,w,x); c[0]['state'][:] = 42
    rows = mod.score_sequence(s,w,x,c)
    assert not rows[0]['primary_passed']
    assert rows[1]['composed_state']['maxrel'] == rows[2]['composed_state']['maxrel'] == 0


def test_rounding_routes_are_diagnostic_and_separate_entry_from_history():
    s = spec(); w = weights(s); w['norm'] = np.array([1.,1.001])
    w['qkv'][:] = [1.,-1.]; x = [np.ones(2)]*3
    c = captures(s,w,x)
    rows = mod.score_sequence(s,w,x,c)
    # History starts at zero, so rounding it cannot affect the first output.
    assert rows[0]['rounding_diagnostics']['history']['effect']['state']['maxrel'] == 0
    assert rows[1]['rounding_diagnostics']['history']['effect']['state']['maxrel'] > 0
    assert rows[0]['rounding_diagnostics']['entry']['effect']['state']['maxrel'] > .005
    assert all(r['primary_passed'] for r in rows)  # diagnostic failures cannot change primary status


@pytest.mark.parametrize('damage', ['norm','qkv','conv','alpha','beta','a','dt','nan','positive_a'])
def test_weights_fail_closed(damage):
    s = spec(); w = weights(s)
    if damage == 'nan': w['conv'][0,0] = np.nan
    elif damage == 'positive_a': w['a'][0] = .1
    else: w[damage] = w[damage][:-1]
    with pytest.raises(ValueError): mod.validate_weights(s,w)


@pytest.mark.parametrize('damage', ['missing','empty','nonfinite','wrong_input','wrong_output'])
def test_sequence_requires_complete_finite_positions(damage):
    s = spec(); w = weights(s); x = [np.array([1., -.5])]*2; c = captures(s,w,x)
    if damage == 'missing': c.pop()
    if damage == 'empty': x=[]; c=[]
    if damage == 'nonfinite': c[-1]['xn'][0]=np.nan
    if damage == 'wrong_input': x[-1]=np.ones(3)
    if damage == 'wrong_output': c[-1]['output']=c[-1]['output'][:-1]
    with pytest.raises(ValueError): mod.score_sequence(s,w,x,c)


def test_each_record_field_is_gated_separately():
    r = np.zeros((6,512)); r[:, 256:384] = 1e6; r[:,384] = .5
    got = r.copy(); got[:,384] += .01
    scores = mod.record_metrics(got,r)
    assert scores['v']['passed'] and not scores['decay']['passed']


def test_entry_reader_rejects_short_or_overwritten_buffer():
    off=SimpleNamespace(A_BYTES=16,A_XN=0)
    raw=f32_to_bf16(np.array([1.,2.],np.float32)).tobytes()+bytes(12)
    np.testing.assert_array_equal(mod.read_entry(raw,off,2),[1.,2.])
    with pytest.raises(ValueError): mod.read_entry(raw[:-1],off,2)
    off.A_XN=15
    with pytest.raises(ValueError): mod.read_entry(raw,off,2)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    s=spec(); s.family='qwen35'; s.num_layers=4; s.layer_types=[LINEAR]*3+[FULL]
    w=weights(s); nch=s.lin_qkv_dim; heads=s.lin_value_heads
    off=SimpleNamespace(A_XN=0,A_QKV=64,A_VEC=64+nch*4,A_O=64+nch*4+heads*512*4,
                        A_BYTES=64+nch*4+heads*640*4,S_ROWS=130,S_HEAD_BYTES=130*128*4,
                        STATE_S_OFF=3*nch*2,STATE_BYTES=3*nch*2+heads*130*128*4)
    meta=dict(hidden=2,layers=4,tokens=3,max_ctx=8,build_key='build',spec_hash='spec')
    (tmp_path/'decode_reference.json').write_text(json.dumps(meta))
    (tmp_path/f'zstate_{LINEAR}.bin').write_bytes(bytes(off.STATE_BYTES))
    x=[np.array([1.,-.5])]*3
    for l in range(3):
        for t,c in enumerate(captures(s,w,x)):
            suffix='' if t==0 else f'_t{t}'
            name=f'y_res{l-1}{suffix}.bin' if l else f'xres{t}.bin'
            x[t].astype('<f4').tofile(tmp_path/name)
            act=(f32_to_bf16(c['xn'].astype(np.float32)).tobytes()+bytes(60)+
                 c['qkv'].astype('<f4').tobytes()+c['records'].astype('<f4').tobytes()+
                 c['output'].astype('<f4').tobytes())
            (tmp_path/f'y_act{l}{suffix}.bin').write_bytes(act)
            (tmp_path/f'y_state{l}{suffix}.bin').write_bytes(c['conv'].tobytes()+c['state'].astype('<f4').tobytes())
    names={'input_layernorm.weight':'norm','linear_attn.ssm_conv1d.weight':'conv',
           'linear_attn.ssm_alpha_proj.bf16.weight':'alpha','linear_attn.ssm_beta_proj.bf16.weight':'beta',
           'linear_attn.ssm_a':'a','linear_attn.ssm_dt.bias':'dt'}
    read=lambda name:w[names[name.split('.',3)[-1]]]
    model=SimpleNamespace(bf16=read,f32=read,matmul_w=lambda *a:w['qkv'])
    monkeypatch.setattr(mod,'spec_from_model_dir',lambda _:s)
    monkeypatch.setattr(mod,'norm_split',lambda _:True)
    monkeypatch.setattr(mod,'load_reference',lambda _:meta)
    monkeypatch.setattr(mod,'manifest',lambda *a,**k:{})
    monkeypatch.setattr(mod,'load_bundle',lambda *a:dict(build_key='build',spec_hash='spec'))
    monkeypatch.setattr(mod,'layout',lambda *a,**k:off)
    monkeypatch.setattr(mod,'Q4NX',lambda _:model)
    return tmp_path,meta


def test_complete_checker_pins_every_weight_and_capture(bundle):
    p,_=bundle; r=mod.check(p,p,p)
    assert len(r['results'])==9 and all(x['primary_passed'] for x in r['results'])
    assert len(r['capture_sha256'])==28 and len(r['weight_sha256'])==21
    assert r['layers']==[0,1,2] and r['tokens']==3


@pytest.mark.parametrize('damage',['missing','residual_size','build','hidden','layers','single_token','initial'])
def test_checker_rejects_invalid_fixture(bundle,damage):
    p,m=bundle
    if damage=='missing':(p/'y_state2_t2.bin').unlink()
    if damage=='residual_size':(p/'y_res1_t2.bin').write_bytes(bytes(7))
    if damage=='build':m['build_key']='bad'
    if damage=='hidden':m['hidden']=3
    if damage=='layers':m['layers']=5
    if damage=='single_token':m['tokens']=1
    if damage=='initial':(p/f'zstate_{LINEAR}.bin').write_bytes(bytes(1))
    with pytest.raises((OSError,ValueError)):mod.check(p,p,p)
