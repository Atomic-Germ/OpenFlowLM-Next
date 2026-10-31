"""Causal prefill checks must not accept missing positions or divergent continuation."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
s=importlib.util.spec_from_file_location('prefill_slice',ROOT/'utilities/prefill-slice/check.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_complete_prompt_and_continuation_coverage():
    assert m.expected_dumps('seq',3)==['seq_p0.bin','seq_p1.bin','seq_p2.bin','seq_t0.bin','seq_t1.bin']
    with pytest.raises(ValueError):m.expected_dumps('seq',0)


def test_logits_require_shape_finite_correlation_and_argmax():
    ref=np.arange(12,dtype=float)
    assert m.logit_score(ref,ref,10)['passed']
    bad=ref.copy();bad[8]=bad[9]+.001
    assert not m.logit_score(bad,ref,10)['passed']
    for bad in [ref[:-1],np.full(12,np.nan),np.zeros(12)]:
        with pytest.raises(ValueError):m.logit_score(bad,ref,10)


def test_continuation_rejects_different_token_despite_matching_logits():
    with pytest.raises(ValueError,match='continuation'):
        m.check_tokens('token 7\ntoken 8\n',[6,8])
    m.check_tokens('prefill 3 tokens\ntoken 6\ntoken 8\n',[6,8])
    with pytest.raises(ValueError):m.check_tokens('token 6\n',[6,8])


def test_read_rejects_partial_nonfinite_and_changed_reference(tmp_path):
    p=tmp_path/'x';np.arange(4,dtype='<f4').tofile(p);h=m.digest(p)
    np.testing.assert_array_equal(m.read(p,4,h),np.arange(4))
    with pytest.raises(ValueError):m.read(p,3)
    np.ones(4,dtype='<f4').tofile(p)
    with pytest.raises(ValueError):m.read(p,4,h)
    np.full(4,np.nan,dtype='<f4').tofile(p)
    with pytest.raises(ValueError):m.read(p,4)


def test_last_residual_adds_both_down_partials():
    from types import SimpleNamespace
    l=SimpleNamespace(AA_BYTES=48,AA_RES=0,AA_OUT2=16,AA_OUT2B=32)
    a=np.arange(12,dtype='<f4').tobytes()
    np.testing.assert_array_equal(m.final_residual(a,l,4),np.array([12,15,18,21]))
    with pytest.raises(ValueError):m.final_residual(a[:-4],l,4)


def test_failed_run_invalidates_prior_success(tmp_path):
    from types import SimpleNamespace
    import json
    (tmp_path/'report.json').write_text('{"passed":true}')
    with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=tmp_path,cli=tmp_path/'missing'))
    assert not json.loads((tmp_path/'report.json').read_text())['passed']


def test_changed_source_is_rejected(tmp_path):
    p=tmp_path/'source';p.write_text('before');hashes={str(p):m.digest(p)}
    p.write_text('after')
    with pytest.raises(ValueError,match='provenance mismatch'):m.verify(hashes)


def test_batched_q8_head_matches_existing_scalar_reference():
    from types import SimpleNamespace
    from q4nx import Q4NX,f32_to_bf16
    rng=np.random.default_rng(8);hidden=512;vocab=64
    chunks=np.zeros((vocab//32*hidden//256,8704),np.uint8)
    chunks[:,:512]=f32_to_bf16(np.full((len(chunks),256),.015625,np.float32)).view(np.uint8)
    chunks[:,512:]=rng.integers(-127,128,(len(chunks),8192),dtype=np.int16).astype(np.int8).view(np.uint8)
    model=SimpleNamespace(raw=lambda name:chunks.tobytes(),chunk_bytes_of=lambda name:8704,hidden=hidden)
    xs=rng.standard_normal((3,hidden)).astype(np.float32)
    want=np.stack([Q4NX.lmhead_logits(model,x,block=2) for x in xs])
    got=m.head_batch(model,xs,vocab)
    np.testing.assert_allclose(got,want,rtol=2e-5,atol=2e-5)
    with pytest.raises(ValueError):m.head_batch(model,xs[:,:-1],vocab)


def test_boundary_prefixes_require_all_three_real_lengths():
    ids=list(range(300))
    cases=m.prompt_cases(ids,True)
    assert list(cases)==[255,256,257]
    assert all(tokens==ids[:n] for n,tokens in cases.items())
    assert m.prompt_cases(ids[:20],False)=={20:ids[:20]}
    for bad in [[],ids[:256]]:
        with pytest.raises(ValueError):m.prompt_cases(bad,True)


def test_causal_snapshots_match_separate_runs_without_aliasing():
    # The decoder deliberately mutates every state buffer, as DeltaNet mutates S.
    def decode(x,state,pos):
        for a in state:a[:]+=x+pos
        return np.array([state[0][0]]),state
    initial=tuple(np.zeros(2) for _ in range(4));xs=np.arange(257)
    rows,states=m.causal_prefixes(xs,initial,(255,256,257),decode)
    for n in (255,256,257):
        state=tuple(np.zeros(2) for _ in range(4))
        for pos,x in enumerate(xs[:n]):want,state=decode(x,state,pos)
        np.testing.assert_array_equal(rows[n-1],want)
        for actual,expected in zip(states[n],state):np.testing.assert_array_equal(actual,expected)
    saved=states[256][1].copy()
    states[255][1][:]=-1
    np.testing.assert_array_equal(states[256][1],saved)
    assert not np.shares_memory(states[257][1],initial[1])


@pytest.mark.parametrize('lengths',[(255,256,258),(0,256,257),(255,255,257),()])
def test_causal_prefixes_reject_invalid_checkpoints(lengths):
    with pytest.raises(ValueError):
        m.causal_prefixes(np.zeros(257),(),lengths,lambda x,s,p:(x,s))


@pytest.mark.parametrize('boundaries',[False,True])
@pytest.mark.parametrize('layers',[4,64])
def test_prepare_uses_each_prefix_state_and_own_prediction(tmp_path,monkeypatch,boundaries,layers):
    import json
    import sys
    from types import SimpleNamespace as NS
    model=tmp_path/'model';kernels=tmp_path/'kernels';model.mkdir();kernels.mkdir()
    for name in ('model.q4nx','config.json','tokenizer.json'):(model/name).write_text('{}')
    for name in ('spec.json','toolchain.json'):(kernels/name).write_text('{}')
    bundle=dict(max_ctx_default=2048,build_key='test',spec_hash='test')
    (kernels/'manifest.json').write_text(json.dumps(bundle))
    original=tmp_path/'original.json'
    original.write_text(json.dumps({str(model/'model.q4nx'):m.digest(model/'model.q4nx')}))
    spec=NS(family='qwen35',hidden=5120,num_layers=64,
            layer_types=(['linear_attention']*3+['full_attention'])*16,real_vocab=32,vocab=32,
            conv_kernel=4,lin_qkv_dim=1,lin_value_heads=1,lin_value_dim=1,
            num_kv_heads=1,head_dim=1)
    ids=[i%32 for i in range(270 if boundaries else 20)]
    monkeypatch.setitem(sys.modules,'tokenizers',NS(Tokenizer=NS(from_file=lambda p:None)))
    monkeypatch.setattr(m,'prompt_ids',lambda *args:ids)
    monkeypatch.setattr(m,'spec_from_model_dir',lambda p:spec)
    monkeypatch.setattr(m,'manifest',lambda *args,**kwargs:bundle)
    monkeypatch.setattr(m,'load_bundle',lambda *args:bundle)
    monkeypatch.setattr(m,'Q4NX',lambda p:NS(matmul_w=lambda *args:None,
                                          embed=lambda t,h:np.full(h,t,np.float32)))
    def decode(model,spec,layer,x,conv,state,k,v,pos):
        assert np.all(state==pos)  # Detect late snapshots and continuation cross-contamination.
        state+=1
        return x+1,conv,state,k,v
    def head(model,spec,xs):
        out=np.zeros((len(xs),32),np.float32)
        for i,x in enumerate(xs):out[i,int(x[0])%32]=1
        return out
    monkeypatch.setattr(m.oracle,'layer_decode',decode)
    monkeypatch.setattr(m,'tail_batch',head)
    out=tmp_path/'out'
    m.prepare(NS(out=out,model=model,kernels=kernels,original=original,
                 message='fixture',boundaries=boundaries,layers=layers))
    for n in ((255,256,257) if boundaries else (20,)):
        d=out/str(n) if boundaries else out
        meta=json.loads((d/'reference.json').read_text())
        expected=(ids[n-1]+layers)%32
        assert meta['layers']==layers
        assert meta['ids']==ids[:n]
        assert meta['expected_tokens']==[expected,(expected+layers)%32]
        assert len(meta['references'])==n+2*layers+1
        m.verify({str(d/name):sha for name,sha in meta['references'].items()})
        np.testing.assert_array_equal(m.read(d/f'ref_cont_res{layers-1}.bin',5120),np.full(5120,expected+layers))


@pytest.mark.parametrize('layers',[0,3,8,65,64.0,'64'])
def test_invalid_depth_is_rejected(layers):
    with pytest.raises(ValueError,match='layers'):m.checked_layers(layers)


def test_full_depth_command_captures_layer_63(tmp_path):
    from types import SimpleNamespace as NS
    meta=dict(layers=64,model='model',kernels='kernels',max_ctx=2048)
    cmd=m.native_command(tmp_path/'cli',tmp_path,meta,NS(AA_BYTES=100),'hybrid',['--gemm-block'])
    assert cmd[cmd.index('--layers')+1]=='64'
    assert cmd[cmd.index('--dump-act')+1]==f'63:0:100:{tmp_path}/hybrid-act.bin'
    assert cmd[cmd.index('--decode-route')+1]=='npu'


def test_reference_coverage_cannot_relabel_slice_as_full_depth():
    meta=dict(layers=64,ids=[1,2],references={name:'hash' for name in
        ['ref_p0.bin','ref_p1.bin','ref_cont_logits.bin']+
        [f'ref_{part}res{i}.bin' for part in ('','cont_') for i in range(4)]})
    with pytest.raises(ValueError,match='reference coverage'):m.check_reference_coverage(meta)
    meta['references'].update({f'ref_{part}res{i}.bin':'hash'
                              for part in ('','cont_') for i in range(64)})
    m.check_reference_coverage(meta)


def test_full_depth_memory_preflight_rejects_insufficient_ram():
    assert m.memory_preflight(64,40*2**30)==40*2**30
    assert m.memory_preflight(4,16*2**30)==16*2**30
    with pytest.raises(RuntimeError,match='35 GiB'):m.memory_preflight(64,16*2**30)


def test_low_memory_run_invalidates_success_without_starting_native(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace as NS
    meta=dict(layers=64,ids=[1],references={name:'unused' for name in
        ['ref_p0.bin','ref_cont_logits.bin']+
        [f'ref_{part}res{i}.bin' for part in ('','cont_') for i in range(64)]})
    (tmp_path/'reference.json').write_text(json.dumps(meta))
    (tmp_path/'report.json').write_text('{"passed":true}')
    preflight=m.memory_preflight
    monkeypatch.setattr(m,'memory_preflight',lambda layers:preflight(layers,16*2**30))
    def unexpected(*args,**kwargs):pytest.fail('native run must not start')
    monkeypatch.setattr(m.subprocess,'run',unexpected)
    with pytest.raises(RuntimeError,match='35 GiB'):
        m.run(NS(out=tmp_path,cli=tmp_path/'cli'))
    report=json.loads((tmp_path/'report.json').read_text())
    assert not report['passed'] and not report['routes']
    assert report['last_residual_layer']==63 and '35 GiB' in report['error']


@pytest.mark.parametrize('wrong_last',[False,True])
def test_full_depth_run_uses_final_layer_reference(tmp_path,monkeypatch,wrong_last):
    import json
    from types import SimpleNamespace as NS
    meta=dict(layers=64,ids=[1],model='model',kernels='kernels',max_ctx=2048,
              vocab=12,real_vocab=10,hidden=4,sources={},references={},expected_tokens=[9,9])
    (tmp_path/'ids.txt').write_text('1\n');meta['ids_sha256']=m.digest(tmp_path/'ids.txt')
    for part in ('','cont_'):
        for i in range(64):
            name=f'ref_{part}res{i}.bin'
            value=np.array([12,15,18,21]) if i==63 else np.full(4,-999)
            if part=='cont_' and i==63 and wrong_last:value+=1
            value.astype('<f4').tofile(tmp_path/name);meta['references'][name]=m.digest(tmp_path/name)
    for name in ('ref_p0.bin','ref_cont_logits.bin'):
        np.arange(12,dtype='<f4').tofile(tmp_path/name);meta['references'][name]=m.digest(tmp_path/name)
    (tmp_path/'reference.json').write_text(json.dumps(meta));(tmp_path/'cli').write_text('fixture')
    monkeypatch.setattr(m,'memory_preflight',lambda layers:40*2**30)
    monkeypatch.setattr(m,'spec_from_model_dir',lambda p:None)
    monkeypatch.setattr(m,'layout',lambda *a,**k:NS(AA_BYTES=48,AA_RES=0,AA_OUT2=16,AA_OUT2B=32))
    monkeypatch.setattr(m,'manifest',lambda *a,**k:None)
    monkeypatch.setattr(m,'load_bundle',lambda *a:None)
    calls=[]
    def native(cmd,stdout,**kwargs):
        assert cmd[cmd.index('--layers')+1]=='64'
        prefix=cmd[cmd.index('--dump-logits')+1]
        act=cmd[cmd.index('--dump-act')+1];assert act.startswith('63:0:48:')
        for name in m.expected_dumps(prefix,1):np.arange(12,dtype='<f4').tofile(name)
        np.arange(12,dtype='<f4').tofile(act.split(':',3)[3])
        stdout.write('token 9\ntoken 9\n');calls.append(cmd)
    monkeypatch.setattr(m.subprocess,'run',native)
    assert m.run(NS(out=tmp_path,cli=tmp_path/'cli'))==int(wrong_last)
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['passed']==(not wrong_last) and len(calls)==3
    assert all(r['passed']==(not wrong_last) for r in report['routes'])
