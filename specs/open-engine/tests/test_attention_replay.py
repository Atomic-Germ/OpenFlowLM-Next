"""Replay must preserve streams, expose softmax state and pin its exact inputs."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[3]/'utilities/attention-replay'
sys.path.insert(0,str(HERE))
import replay_common as mod
from q4nx import f32_to_bf16


def geom():
    return dict(tokens=2,heads=4,kv_heads=2,dim=32,cores=2,local_heads=2,mls=32,
                ea=128,qw=128,kw=64,input_row=1792,qblocks=4,oblocks=2,
                debug_blocks=8,debug_core=1024)


def test_stream_packing_preserves_all_regions():
    g=geom();meta=bytes([17])*128;ptab=bytes([19])*256
    cs=[{n:np.full((4 if n in ('q','gate') else 2,32),i+j,np.float64)
         for j,n in enumerate(('q','k','v','gate'))} for i in range(2)]
    data=mod.pack_input(cs,meta,ptab,g)
    assert len(data)==3584
    assert data[:256]==meta+ptab[:128]
    np.testing.assert_array_equal(np.frombuffer(data[256:768],np.float32),cs[0]['q'].ravel())
    np.testing.assert_array_equal(np.frombuffer(data[1280:1792],np.float32),cs[0]['gate'].ravel())
    with pytest.raises(ValueError):mod.pack_input(cs,meta[:-1],ptab,g)


def test_debug_reads_q_pairs_accumulators_and_padded_ml(tmp_path):
    g=geom();a=np.zeros((2,2,1024),np.uint8)
    for t in range(2):
        for c in range(2):
            a[t,c,:512]=np.frombuffer(f32_to_bf16(np.full(256,1+t,np.float32)).tobytes(),np.uint8)
            a[t,c,512:768]=np.frombuffer(np.full(64,3+c,np.float32).tobytes(),np.uint8)
            a[t,c,768:1024]=np.frombuffer(np.full(64,5+t,np.float32).tobytes(),np.uint8)
    p=tmp_path/'debug';p.write_bytes(a.tobytes());r=mod.read_debug(p,g)
    np.testing.assert_array_equal(r['q_scaled'][0],2.)
    np.testing.assert_array_equal(r['q_scaled'][1],4.)
    np.testing.assert_array_equal(r['oacc'][0,2:],4.)
    assert r['ml'].shape==(2,2,2,32)
    p.write_bytes(a.tobytes()[:-1])
    with pytest.raises(ValueError):mod.read_debug(p,g)


def test_conditional_attention_uses_actual_kv_history_and_scaled_q():
    q=np.ones((2,4,2));kv=np.zeros((2,2,2,2));kv[0,1]=[[2,4],[6,8]];kv[1,1]=[[10,12],[14,16]]
    out=mod.conditional_attention(q,kv)
    np.testing.assert_array_equal(out[0],np.repeat(kv[0,1],2,axis=0))
    np.testing.assert_array_equal(out[1],np.repeat((kv[0,1]+kv[1,1])/2,2,axis=0))


def test_corrupt_hash_and_missing_coverage_refused(tmp_path):
    f=tmp_path/'a';f.write_bytes(b'a');p=tmp_path/'audit.json'
    p.write_text(json.dumps({'sha256':{str(f.resolve()):mod.digest(f)}}))
    mod.verify(tmp_path,[f])
    with pytest.raises(ValueError,match='coverage'):mod.verify(tmp_path,[f,tmp_path/'missing'])
    f.write_bytes(b'b')
    with pytest.raises(ValueError,match='hash'):mod.verify(tmp_path,[f])


def test_nonfinite_debug_and_head_copy_disagreement_refused(tmp_path):
    g=geom();a=np.zeros((2,2,1024),np.uint8);a[0,1,0:2]=[0x80,0x3f]
    p=tmp_path/'debug';p.write_bytes(a.tobytes())
    with pytest.raises(ValueError,match='copies'):mod.read_debug(p,g)
    a[:]=0;a[0,0,512:516]=np.frombuffer(np.float32(np.nan).tobytes(),np.uint8);p.write_bytes(a.tobytes())
    with pytest.raises(ValueError,match='finite'):mod.read_debug(p,g)


def test_requested_mode_must_match_build_metadata():
    with pytest.raises(ValueError,match='mode'):mod.validate_build(dict(mode='baseline',tokens=3), 'fp32',3)
    with pytest.raises(ValueError,match='tokens'):mod.validate_build(dict(mode='fp32',tokens=2), 'fp32',3)


def test_stale_score_removed_before_validation(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('attn_replay_check',HERE/'check.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    out=tmp_path/'out';out.mkdir();(out/'score.json').write_text('{"passed":true}')
    monkeypatch.setattr('sys.argv',['check','--model-dir',str(tmp_path/'model'),'--kernel-dir',str(tmp_path/'kernels'),
        '--fixture',str(tmp_path/'fixture'),'--out',str(out),'--build',str(tmp_path/'build'),'--mode','baseline'])
    with pytest.raises((ValueError,OSError)):c.main()
    assert not (out/'score.json').exists()


def score_fixture():
    spec=importlib.util.spec_from_file_location('attn_replay_score',HERE/'check.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    g=geom();s=SimpleNamespace(norm_eps=1e-6,rotary_dim=0,rope_inv_freq=lambda:np.zeros(0))
    q=np.ones((2,4,32));k=np.ones((2,2,32));v=np.ones((2,2,32));v[1]*=2
    gate=np.zeros_like(q);nw=np.ones(32)
    from attention_heads import attention_reference
    output=attention_reference(q,k,v,gate,nw,nw,1e-6,0,np.zeros(0))
    cs=[dict(q=q[t],k=k[t],v=v[t],gate=gate[t],og=output[t]) for t in range(2)]
    kv=np.stack([k/np.sqrt(1+1e-6),v],axis=1)
    qs=q/np.sqrt(1+1e-6)/np.sqrt(32)
    dbg=dict(q_scaled=qs,oacc=mod.conditional_attention(qs,kv),ml=np.ones((2,2,2,32)))
    return c,[s,g,cs,nw,nw,output,kv,dbg]


def test_exact_primary_and_captured_boundaries_pass():
    c,args=score_fixture();r=c.score_layer(*args)
    assert r['passed'] and r['production_output_byte_identical']
    assert all(x['passed'] for x in r['diagnostics']['softmax_pv_from_captured_qkv'])
    assert all(x['passed'] for x in r['diagnostics']['output_from_captured_kv'])


def test_captured_kv_diagnostic_cannot_hide_primary_error():
    c,args=score_fixture();args[6][:,1]+=2
    cond=mod.conditional_attention(args[7]['q_scaled'],args[6]);args[7]['oacc']=cond
    args[5]=cond*.5
    r=c.score_layer(*args)
    assert not r['passed'] and not r['production_output_byte_identical']
    assert all(x['passed'] for x in r['diagnostics']['output_from_captured_qkv'])


def test_zero_softmax_sum_refused():
    c,args=score_fixture();args[7]['ml'][:,:,1]=0
    with pytest.raises(ValueError,match='sum'):c.score_layer(*args)


def test_different_build_modes_and_token_counts_get_distinct_keys():
    from build import source_hash
    from recipes.load import load_spec
    s=load_spec(HERE.parents[1]/'open_kernels/recipes/specs/qwen35-27b.json')
    assert len({source_hash(s,0,3),source_hash(s,1,3),source_hash(s,1,4)})==3
