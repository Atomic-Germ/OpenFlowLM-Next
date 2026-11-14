"""Audit exact combined captures and conditioned causal boundaries independently."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_combined_sublayer_build',HERE/'build.py')
builder=importlib.util.module_from_spec(s);s.loader.exec_module(builder)
runner=builder.runner;base=runner.base;trace=runner.trace;digest=runner.digest
sub=builder.load('_combined_sublayer_math',builder.ROOT/'utilities/prefill-sublayer-trace/check.py')
linear=builder.load('_combined_linear_math',builder.ROOT/'utilities/prefill-prefix-sublayers/check.py')
rb=linear.rb
TARGETS=((63,1),(42,17),(52,17))
COMMON=('input','res','norm','upgate','h','down')
LINEAR=('xn','qkv','z','og','state_before','state_after','proj')
ATTENTION=('xn','q','k','v','gate','og','kv_after','proj')


def expected_captures():
    names={f'sequential_L{l}_P{p}_act.bin' for l,p in TARGETS}
    names.update(f'{r}_L{l}_P{p}_{k}.bin' for r in ('hybrid','block_major') for l,p in TARGETS
                 for k in COMMON+(ATTENTION if l==63 else LINEAR))
    return names


def inventory(run):
    if {p.name for p in run.glob('*_L*_P*.bin')}!=expected_captures():
        raise ValueError('incomplete or unexpected sublayer inventory')


def head_metrics(got,ref,width):
    got,ref=np.asarray(got,np.float64),np.asarray(ref,np.float64)
    if got.shape!=ref.shape or got.ndim!=2 or not width or got.shape[1]%width:
        raise ValueError('invalid per-head geometry')
    return [dict(position=p,head=h,metrics=linear.metrics(got[p,h*width:(h+1)*width],ref[p,h*width:(h+1)*width]))
            for p in range(len(got)) for h in range(got.shape[1]//width)]


def attention_reference(values,qn,kn,inv_freq,nh,kvh,hd,eps):
    kv=np.zeros((len(values['q']),4*kvh*hd),np.float64)
    # ModelSpec returns a list; multiplying it by position 0 would empty it.
    inv=np.asarray(inv_freq,np.float64)
    og=rb.attention_block(values['q'],values['k'],values['v'],values['gate'],
        qn,kn,inv,kv,0,nh=nh,kvh=kvh,hd=hd,eps=eps,kv_pair=True)
    return og,kv


def context(baseline,build):
    original=json.loads((baseline/'report.json').read_text())
    audit=json.loads((baseline/'audit.json').read_text())
    if not original['complete'] or not audit['complete'] or audit['report_sha256']!=digest(baseline/'report.json'):
        raise ValueError('baseline has no complete independent audit')
    record,meta,offsets=runner.preparation(Path(original['prepared']))
    if original['preparation_sha256']!=digest(Path(original['prepared'])/'preparation.json'):
        raise ValueError('baseline preparation changed')
    b=runner.prep.verify_build(build,'combined-sublayer')
    if digest(b['parent_build'])!=b['parent_build_sha256'] or Path(b['parent_build']).parent!=Path(record['builds']['combined']['directory']):
        raise ValueError('capture build does not derive from measured combined candidate')
    # Validate retained baseline files before allowing a new model load.
    measured=runner.audit_variant(baseline/'combined',Path(original['prepared'])/'references',meta,offsets)
    if any(original['variants']['combined'].get(k)!=v for k,v in measured.items()):
        raise ValueError('baseline captures no longer match audited measurements')
    return original,b,meta,offsets


def boundaries(model,spec,offsets,run,route,l,p):
    n=20;hid=spec.hidden;pre=f'model.layers.{l}.'
    def path(k):return run/f'{route}_L{l}_P{p}_{k}.bin'
    def read(k,w):return base.read(path(k),n*w).reshape(n,w)
    def weight(name,o,i):return model.matmul_w(pre+name,o,i).astype(np.float64)
    inputs=trace.read_trace(run/f'{route}_res{l-1}.bin',n+1,hid)[:n]
    xn=read('xn',hid)
    exact_norm=base.oracle.rms(inputs,spec.norm_eps)*model.bf16(pre+'input_layernorm.weight')
    observed={'norm':xn};expected={'norm':exact_norm};heads={};selected={}
    if l==63:
        qw=spec.num_heads*spec.head_dim;kvw=spec.num_kv_heads*spec.head_dim
        qg=weight('self_attn.q_proj.weight',2*qw,hid)
        weights={'q':qg[:qw],'gate':qg[qw:],
                 'k':weight('self_attn.k_proj.weight',kvw,hid),
                 'v':weight('self_attn.v_proj.weight',kvw,hid)}
        values={k:read(k,w.shape[0]) for k,w in weights.items()}
        ow=weight('self_attn.o_proj.weight',hid,qw)
        og=read('og',qw)
        ideal,kv=attention_reference(values,
            model.bf16(pre+'self_attn.q_norm.weight'),model.bf16(pre+'self_attn.k_norm.weight'),
            spec.rope_inv_freq(),spec.num_heads,spec.num_kv_heads,spec.head_dim,spec.norm_eps)
        raw=path('kv_after').read_bytes()
        if len(raw)!=n*4*kvw*2:raise ValueError('wrong paired KV capture extent')
        captured=(np.frombuffer(raw,'<u2').astype(np.uint32)<<16).view(np.float32).astype(np.float64).reshape(n,4*kvw)
        # Low halves can straddle BF16 rounding boundaries; compare effective paired values.
        selected['paired_k']=linear.metrics(captured[:,:kvw]+captured[:,kvw:2*kvw],kv[:,:kvw]+kv[:,kvw:2*kvw])
        selected['paired_v']=linear.metrics(captured[:,2*kvw:3*kvw]+captured[:,3*kvw:],kv[:,2*kvw:3*kvw]+kv[:,3*kvw:])
        width=spec.head_dim
    else:
        vw=spec.lin_value_heads*spec.lin_value_dim
        weights={'qkv':weight('linear_attn.qkv_proj.weight',spec.lin_qkv_dim,hid),
                 'z':weight('self_attn.gate_proj.weight',vw,hid)}
        values={k:read(k,w.shape[0]) for k,w in weights.items()}
        ow=weight('linear_attn.ssm_out_proj.weight',hid,vw);og=read('og',vw)
        conv,state=linear.states(path('state_before').read_bytes(),path('state_after').read_bytes(),offsets,spec)
        if not np.array_equal(conv,values['qkv'][-(spec.conv_kernel-1):]):
            raise ValueError('conv carry differs from captured QKV tail')
        constants=[model.bf16(pre+'linear_attn.'+name).astype(np.float64) for name in
                   ('ssm_conv1d.weight','ssm_alpha_proj.bf16.weight','ssm_beta_proj.bf16.weight')]
        constants[1]=constants[1].T;constants[2]=constants[2].T
        constants += [model.f32(pre+'linear_attn.ssm_a').astype(np.float64),
                      model.f32(pre+'linear_attn.ssm_dt.bias').astype(np.float64),
                      model.bf16(pre+'linear_attn.ssm_norm.weight').astype(np.float64)]
        ideal_state=np.zeros_like(state)
        ideal=rb.deltanet_block(values['qkv'],values['z'],xn,*constants,np.zeros_like(conv),ideal_state,
                key_heads=spec.lin_key_heads,value_heads=spec.lin_value_heads,head_dim=spec.lin_value_dim,
                eps=spec.norm_eps,conv_f32=True)
        selected['block_end_state']=linear.metrics(state[:,:spec.lin_value_dim],ideal_state[:,:spec.lin_value_dim])
        width=spec.lin_value_dim
    for key,w in weights.items():
        for source,x in [('input',exact_norm),('norm',xn)]:
            name=key+'_from_'+source;observed[name]=values[key];expected[name]=x@w.T
            heads[name]=head_metrics(values[key],expected[name],width)
    observed['og_from_captured_prefix']=og;expected['og_from_captured_prefix']=ideal
    heads['og_from_captured_prefix']=head_metrics(og,ideal,width)
    proj=base.read(path('proj'),hid);res=base.read(path('res'),hid)
    if not np.array_equal(res.astype(np.float32),inputs[p].astype(np.float32)+proj.astype(np.float32)):
        raise ValueError('input/projection do not reconstruct pre-FFN residual')
    selected['output_projection']=linear.metrics(proj,og[p]@ow.T)
    rows=[dict(position=t,metrics={k:linear.metrics(observed[k][t],v[t]) for k,v in expected.items()}) for t in range(n)]
    return dict(route=route,layer=l,position=p,rows=rows,heads=heads,selected=selected,
                residual_reconstruction_exact=True,state_position=n-1)


def check(a):
    result=dict(capture_valid=False,conditioned_ffn_passed=False,conditioned_boundaries_passed=False,model_acceptance=False)
    try:
        run=a.run.resolve();saved=json.loads((run/'report.json').read_text())
        if saved['runner_sha256']!=digest(HERE/'run.py'):
            raise ValueError('capture runner changed')
        baseline=Path(saved['baseline']);build=Path(saved['build'])
        original,b,meta,offsets=context(baseline,build)
        if not saved['complete'] or saved['cli_sha256']!=b['cli_sha256'] or saved['baseline_report_sha256']!=digest(baseline/'report.json'):
            raise ValueError('incomplete captures or changed run identity')
        refs=Path(original['prepared'])/'references';n=len(meta['ids']);hid=meta['hidden']
        if n!=20 or hid!=5120 or meta['layers']!=64:raise ValueError('requires complete 20-token full-depth prompt')
        identity=runner.require_identity(baseline/'combined',run,n,meta['layers'])
        measured=runner.audit_variant(run,refs,meta,offsets)
        if measured!=saved['measurements']:raise ValueError('reported full-depth gates differ')
        inventory(run);hashes={name:digest(run/name) for name in expected_captures()}
        if hashes!=saved['captures']:raise ValueError('sublayer captures changed')
        pairs={name:hashes[name]==hashes[name.replace('hybrid','block_major',1)] for name in hashes if name.startswith('hybrid_')}
        if not all(pairs.values()):raise ValueError('hybrid sublayer schedules differ')
        spec=base.spec_from_model_dir(Path(meta['model']))
        model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=hid;model.native_q8=lambda name:False
        plain=model.matmul_w;cache={}
        def cached(name,o,i):
            if name not in cache:cache[name]=plain(name,o,i)
            return cache[name]
        model.matmul_w=cached;cases=[];projection=[]
        for l,p in TARGETS:
            cache.clear();pre=f'model.layers.{l}.'
            ref_inputs=base.read(refs/f'ref_res{l-1}.bin',n*hid).reshape(n,hid)[:p+1].astype(np.float32)
            independent_res=sub.causal_res(model,spec,l,ref_inputs)
            independent_out=independent_res+base.oracle.ffn(model,spec,l,independent_res)
            expected_out=base.read(refs/f'ref_res{l}.bin',n*hid).reshape(n,hid)[p]
            if independent_out.astype('<f4').tobytes()!=expected_out.astype('<f4').tobytes():
                raise ValueError('independent causal output differs from retained reference')
            nw=model.bf16(pre+'post_attention_layernorm.weight').astype(np.float64)
            up=cached(pre+'mlp.up_proj.weight',spec.intermediate,hid).astype(np.float64)
            gate=cached(pre+'mlp.gate_proj.weight',spec.intermediate,hid).astype(np.float64)
            down=cached(pre+'mlp.down_proj.weight',hid,spec.intermediate).astype(np.float64)
            for route in runner.ROUTES:
                inputs=trace.read_trace(run/f'{route}_res{l-1}.bin',n+1,hid)[:p+1].astype(np.float32)
                output=trace.read_trace(run/f'{route}_res{l}.bin',n+1,hid)[p]
                c=sub.selected_captures(run,route,l,p,meta,offsets)
                if 'input' in c and not np.array_equal(c['input'],inputs[-1]):raise ValueError('captured input differs')
                if np.any(np.abs(output-(c['res']+c['down']))>4*np.finfo(np.float32).eps*c['sum_bound']):
                    raise ValueError('FFN residual/down do not reconstruct output')
                ideal=sub.causal_res(model,spec,l,inputs)
                case=dict(route=route,layer=l,position=p,cpu_reference_exact=True,
                    input=sub.metric(inputs[-1],ref_inputs[-1]),pre_ffn=sub.metric(c['res'],independent_res),
                    output=sub.metric(output,independent_out),ideal_state_from_native_input=sub.metric(c['res'],ideal),
                    ffn=sub.ffn_metrics(c,nw,up,gate,down,spec.norm_eps),
                    ffn_propagation=sub.output_diagnostics(output,c['res'],base.oracle.ffn(model,spec,l,c['res']),independent_out))
                cases.append(case);print(json.dumps(case),flush=True)
            del up,gate,down;cache.clear()
            for route in ('hybrid','block_major'):
                projection.append(boundaries(model,spec,offsets,run,route,l,p))
                print('projection boundaries complete',route,l,p,flush=True)
            cache.clear()
        checks=[m for c in projection for m in [*c['selected'].values(),
                *(m for row in c['rows'] for m in row['metrics'].values()),
                *(head['metrics'] for heads in c['heads'].values() for head in heads)]]
        files=[Path(__file__),Path(sub.__file__),Path(linear.__file__),Path(rb.__file__),
               Path(base.oracle.__file__),builder.ROOT/'open_kernels/model/ffn_front.py',
               run/'report.json',baseline/'audit.json',build/'build-evidence.json']
        result.update(capture_valid=True,conditioned_ffn_passed=all(m['passed'] for c in cases for m in c['ffn'].values()),
            conditioned_boundaries_passed=all(m['passed'] for m in checks),
            cases=cases,projection_cases=projection,baseline_identity=identity,hybrid_identity=pairs,
            captures=hashes,full_depth_passed=measured['passed'],
            evidence={str(p.resolve()):digest(p) for p in files},
            scope='conditioned diagnostics; captured-prefix history is not a replacement independent model reference')
    except Exception as error:
        result.update(capture_valid=False,error=str(error));raise
    finally:a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['conditioned_ffn_passed'] and result['conditioned_boundaries_passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','out'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(check(p.parse_args()))
