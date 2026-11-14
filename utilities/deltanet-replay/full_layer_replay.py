"""Forced-input full-lx replay with causal state and independent residual reference."""
import argparse,json,re,sys
from pathlib import Path
import numpy as np
from full_layer_common import ROOT,geometry,digest,source_hash,verify_generated
from build_full_layer import verify_dependencies
from recurrence_replay import read_f32
from decode_bundle import load_bundle
from decode_reference import load_reference,sfx
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.spec import LINEAR
import deltanet_state as ds
from ffn_front import metric as residual_metric


def exact_capture(a,b,size):
    left,right=a.read_bytes(),b.read_bytes()
    if len(left)!=size or len(right)!=size:raise ValueError('wrong capture size')
    return left==right


def config(bundle,g,layers,tokens,fixture,pools,out,build):
    paths=[Path(p).resolve() for p in (fixture,pools,out,build)];fixture,pools,out,build=paths
    b=bundle['layer_types'][LINEAR]['buffers']
    lines=['device',f'xclbin lx {build}/final.xclbin',f'kernelx lx lx {build}/insts.bin',
           f'buf pool {bundle["layout"]["pool_bytes"]}',f'buf consts {b["consts"]}',
           f'buf xres {g["hidden"]*4}',f'buf act {g["act_bytes"]}',f'buf state {g["state_bytes"]}']
    for layer in layers:
        lines += [f'load pool {pools}/pool_L{layer}.bin',f'load consts {fixture}/consts_{layer}.bin',
                  f'load state {out}/zero.bin',f'load act {out}/zero_act.bin']
        for t in range(tokens):
            suffix=sfx(t);inp=fixture/(f'y_res{layer-1}{suffix}.bin' if layer else f'xres{t}.bin')
            lines += [f'load xres {inp}','run lx pool xres consts state act',
                      f'dump act {out}/y_act{layer}{suffix}.bin {g["act_bytes"]}',
                      f'dump state {out}/y_state{layer}{suffix}.bin {g["state_bytes"]}',
                      f'dump xres {out}/y_res{layer}{suffix}.bin {g["hidden"]*4}']
    return '\n'.join(lines)+'\n'


def checked_files(hashes):
    if not hashes:raise ValueError('empty hash coverage')
    for n,h in hashes.items():
        if digest(Path(n))!=h:raise ValueError('hash mismatch: '+n)


def context(model,kernels,fixture,build):
    spec=spec_from_model_dir(model);meta=load_reference(fixture)
    bundle=load_bundle(kernels,manifest(spec,max_ctx=meta['max_ctx']))
    if any(bundle[k]!=meta[k] for k in ('build_key','spec_hash')) or meta['tokens']<2:raise ValueError('invalid fixture identity')
    b=json.loads((build/'build.json').read_text());g=geometry(spec)
    if b['geometry']!=g or b['production_build_key']!=bundle['build_key'] or b['source_hash']!=source_hash(spec,b['shared']):
        raise ValueError('invalid private build identity')
    if set(b['sha256'])!={'final.xclbin','insts.bin'}:raise ValueError('incomplete binary hashes')
    checked_files({str(build/n):h for n,h in b['sha256'].items()})
    if b['mode']=='control':
        if any(digest(build/n)!=digest(kernels/'lx'/n) for n in b['sha256']):raise ValueError('changed control')
    elif b['mode']=='factored':
        if verify_generated(build/'generated',spec,b['shared'])!=b['generated_sha256']:raise ValueError('changed sources')
        if digest(build/'full_lx.py')!=b['design_sha256'] or verify_dependencies(build)!=b['dependency_sha256']:raise ValueError('changed compilation')
    else:raise ValueError('invalid build mode')
    return spec,meta,bundle,b,g


def prepare(model,kernels,fixture,pools,build,out):
    if any(out.resolve()==p.resolve() or out.resolve().is_relative_to(p.resolve()) for p in (model,kernels,fixture,pools,build)):
        raise ValueError('separate private replay output required')
    out.mkdir(parents=True,exist_ok=True);(out/'audit.json').unlink(missing_ok=True);(out/'score.json').unlink(missing_ok=True)
    spec,meta,bundle,b,g=context(model,kernels,fixture,build)
    layers=[l for l in range(meta['layers']) if spec.layer_types[l]==LINEAR]
    if not layers:raise ValueError('no linear layers')
    for l in layers:
        for t in range(meta['tokens']):
            for k in ('act','state','res'):(out/f'y_{k}{l}{sfx(t)}.bin').unlink(missing_ok=True)
    (out/'zero.bin').write_bytes(bytes(g['state_bytes']));(out/'zero_act.bin').write_bytes(bytes(g['act_bytes']))
    (out/'run.cfg').write_text(config(bundle,g,layers,meta['tokens'],fixture,pools,out,build))
    files=[build/n for n in ('build.json','final.xclbin','insts.bin')]+[model/'model.q4nx',model/'config.json',
           fixture/'decode_reference.json',out/'zero.bin',out/'zero_act.bin',out/'run.cfg',Path(__file__),
           ROOT/'open_kernels/model/replica_qwen35.py',ROOT/'open_kernels/model/q4nx.py',ROOT/'open_kernels/model/deltanet_state.py']
    for l in layers:
        files += [pools/f'pool_L{l}.bin',fixture/f'consts_{l}.bin']
        for t in range(meta['tokens']):
            inp=fixture/(f'y_res{l-1}{sfx(t)}.bin' if l else f'xres{t}.bin');read_f32(inp,(spec.hidden,))
            files += [inp]+[fixture/f'y_{k}{l}{sfx(t)}.bin' for k in ('act','state','res')]
    audit=dict(mode=b['mode'],layers=layers,tokens=meta['tokens'],build_key=bundle['build_key'],
               model=str(model),kernels=str(kernels),fixture=str(fixture),pools=str(pools),build=str(build),
               sha256={str(p.resolve()):digest(p) for p in files})
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('Prepared',len(layers)*meta['tokens'],'full-layer steps',flush=True)


def verified(out):
    a=json.loads((out/'audit.json').read_text());checked_files(a['sha256'])
    spec,meta,bundle,b,g=context(*(Path(a[n]) for n in ('model','kernels','fixture','build')))
    layers=[l for l in range(meta['layers']) if spec.layer_types[l]==LINEAR]
    if a['layers']!=layers or a['tokens']!=meta['tokens'] or a['mode']!=b['mode'] or a['build_key']!=bundle['build_key']:
        raise ValueError('changed replay coverage')
    expected=config(bundle,g,layers,meta['tokens'],Path(a['fixture']),Path(a['pools']),out,Path(a['build']))
    if (out/'run.cfg').read_text()!=expected:raise ValueError('changed replay program')
    return a,spec,meta,bundle,b,g


def reference(out,refs):
    from q4nx import Q4NX
    import replica_qwen35 as replica
    a,spec,meta,bundle,b,g=verified(out);refs.mkdir(parents=True,exist_ok=True);(refs/'reference.json').unlink(missing_ok=True)
    q=Q4NX(Path(a['model'])/'model.q4nx');patterns=[re.compile(re.escape(op['tensor']).replace(r'\{l\}',r'\d+'))
        for d in bundle['layer_types'].values() for op in d['pack']['pool']+d['pack']['consts'] if op.get('op')=='q8_perm']
    q.native_q8=lambda n:any(p.fullmatch(n) for p in patterns)
    class CachedLayer:
        def __init__(self):self.cache={}
        def matmul_w(self,*args):
            if args not in self.cache:self.cache[args]=q.matmul_w(*args)
            return self.cache[args]
        def bf16(self,*args):return q.bf16(*args)
        def f32(self,*args):return q.f32(*args)
    hashes={};fixture=Path(a['fixture'])
    for l in a['layers']:
        weights=CachedLayer();conv=np.zeros((spec.conv_kernel-1,spec.lin_qkv_dim));state=np.zeros((48,128,128))
        for t in range(a['tokens']):
            inp=fixture/(f'y_res{l-1}{sfx(t)}.bin' if l else f'xres{t}.bin');x=read_f32(inp,(spec.hidden,)).astype(np.float64)
            y,conv,state,_,_=replica.layer_decode(weights,spec,l,x,conv,state,None,None,t)
            ds.finite(y);p=refs/f'ref_res{l}{sfx(t)}.bin';y.astype('<f8').tofile(p);hashes[p.name]=digest(p)
        del weights
        print('Reference layer',l,flush=True)
    # Pin shared inputs independently of a particular control/candidate output path.
    shared={p:h for p,h in a['sha256'].items() if not Path(p).is_relative_to(out) and not Path(p).is_relative_to(Path(a['build']))}
    checked_files(shared)
    (refs/'reference.json').write_text(json.dumps(dict(layers=a['layers'],tokens=a['tokens'],build_key=a['build_key'],
        input_sha256=shared,sha256=hashes,arithmetic='existing replica_qwen35, mixed FP64/FP32, independent causal state'),indent=2)+'\n')


def score(out,refs,single_result,single_run):
    a,spec,meta,bundle,b,g=verified(out);fixture=Path(a['fixture']);off=layout(spec,max_ctx=meta['max_ctx'])
    rm=json.loads((refs/'reference.json').read_text());single=json.loads(single_result.read_text())
    shared={p:h for p,h in a['sha256'].items() if not Path(p).is_relative_to(out) and not Path(p).is_relative_to(Path(a['build']))}
    if rm['input_sha256']!=shared or any(rm[k]!=a[k] for k in ('layers','tokens','build_key')):raise ValueError('reference identity mismatch')
    expected={f'ref_res{l}{sfx(t)}.bin' for l in a['layers'] for t in range(a['tokens'])}
    if set(rm['sha256'])!=expected:raise ValueError('incomplete residual references')
    checked_files({str(refs/n):h for n,h in rm['sha256'].items()})
    if single['mode']!='factored' or not single['passed'] or single['build']['production_build_key']!=a['build_key']:
        raise ValueError('unvalidated isolated candidate')
    rows=[];hashes={};entry_dtype,conv_dtype=ds.linear_dtypes(bundle)
    for l in a['layers']:
        captures=[];base=[]
        for t in range(a['tokens']):
            c=ds.read_boundaries((out/f'y_act{l}{sfx(t)}.bin').read_bytes(),(out/f'y_state{l}{sfx(t)}.bin').read_bytes(),spec,off,conv_dtype=conv_dtype)
            captures.append(c)
            base.append(ds.read_boundaries((fixture/f'y_act{l}{sfx(t)}.bin').read_bytes(),(fixture/f'y_state{l}{sfx(t)}.bin').read_bytes(),spec,off,conv_dtype=conv_dtype))
        recurrent=ds.compare_sequence(captures,conv_dtype=conv_dtype)
        for t,(c,orig,r) in enumerate(zip(captures,base,recurrent)):
            exact=True
            for kind,size in [('act',g['act_bytes']),('state',g['state_bytes']),('res',spec.hidden*4)]:
                p=out/f'y_{kind}{l}{sfx(t)}.bin';hashes[p.name]=digest(p)
                exact &= exact_capture(p,fixture/p.name,size)
            inputs_same=all(c[k].tobytes()==orig[k].tobytes() for k in ('records','qkv','conv'))
            isolated_same=True
            if b['mode']=='factored':
                for h in range(48):
                    for field,values in [('state',c['state'][h]),('output',c['output'][h])]:
                        n=f'L{l}_H{h}_T{t}-{field}.bin';p=single_run/n
                        if digest(p)!=single['capture_sha256'][n]:raise ValueError('isolated capture changed')
                        isolated_same &= values.tobytes()==p.read_bytes()
            raw=(refs/f'ref_res{l}{sfx(t)}.bin').read_bytes()
            if len(raw)!=spec.hidden*8:raise ValueError('invalid reference length')
            ref=np.frombuffer(raw,'<f8');ds.finite(ref)
            residual=residual_metric(read_f32(out/f'y_res{l}{sfx(t)}.bin',(spec.hidden,)).astype(np.float64),ref)
            ok=inputs_same and residual['passed'] and (exact if b['mode']=='control' else isolated_same and r['passed'] and r['head_diagnostics_passed'])
            rows.append(dict(layer=l,token=t,passed=ok,control_identical=exact,inputs_identical=inputs_same,
                             isolated_identical=isolated_same if b['mode']=='factored' else None,residual=residual,recurrence=r))
    result=dict(mode=b['mode'],passed=all(r['passed'] for r in rows),layers=a['layers'],tokens=a['tokens'],build_key=a['build_key'],
                scope='forced incoming production residuals; own causal full-layer state; not full-model propagation',
                evidence_sha256={str(p):digest(p) for p in (out/'audit.json',refs/'reference.json',single_result)},
                capture_sha256=hashes,results=rows)
    (out/'score.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS' if result['passed'] else 'FAIL',len(rows),'full-layer cases',flush=True);return 0 if result['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','reference','score'])
    for n in ('model','kernels','fixture','pools','build','out','refs','single-result','single-run'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    needed={'prepare':['model','kernels','fixture','pools','build','out'],'reference':['out','refs'],'score':['out','refs','single_result','single_run']}[a.stage]
    for n in needed:
        if getattr(a,n) is None:p.error('missing --'+n.replace('_','-'))
        setattr(a,n,getattr(a,n).resolve())
    if a.stage=='prepare':prepare(*(getattr(a,n) for n in needed))
    elif a.stage=='reference':reference(a.out,a.refs)
    else:return score(a.out,a.refs,a.single_result,a.single_run)
    return 0


if __name__=='__main__':raise SystemExit(main())
