"""Replay the complete L63 attention prefix and compose corrected projections with FFN."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


down=load('_attention_down',ROOT/'utilities/prefill-down-replay/check.py')
bridge=load('_attention_bridge',Path(__file__).with_name('bridge.py'))
attr,prefix,paired,base,proj=down.attr,down.prefix,down.paired,down.base,down.proj
host=prefix.host;digest,verify=down.digest,down.verify
ROWS,T,P=20,256,1
ROLES=(('q',0,6144),('k',6144,7168),('v',7168,8192),('gate',8192,14336))
SHAPES=dict(entry=(14336,5120),out=(5120,6144),up_gate=(34816,5120),down=(5120,17408))


def native_vector(values):
    original=np.asarray(values)
    with np.errstate(over='raise',invalid='raise'):result=np.asarray(original,np.float32)
    if original.ndim!=1 or not original.size or not np.isfinite(result).all() or not np.array_equal(original,result):
        raise ValueError('native capture must be losslessly representable in FP32')
    return np.ascontiguousarray(result)


def split(x):
    return {role:np.ascontiguousarray(x[:,lo:hi]) for role,lo,hi in ROLES}


def entry_scores(control,candidate,reference):
    arrays=[np.asarray(v,np.float64) for v in (control,candidate,reference)]
    if any(v.shape!=(ROWS,14336) or not np.isfinite(v).all() for v in arrays):
        raise ValueError('complete finite attention prefix required')
    rows=[]
    for role,lo,hi in ROLES:
        for t in range(ROWS):
            for h,start in enumerate(range(lo,hi,256)):
                c,p,r=[v[t,start:start+256] for v in arrays]
                rows.append(dict(role=role,position=t,head=h,control=proj.metric(c,r),candidate=proj.metric(p,r)))
    return dict(heads=rows,passed=all(r['candidate']['passed'] for r in rows),
                improved=all(all(r['candidate'][k]<r['control'][k] for k in ('maxrel','rel_fro')) for r in rows))


def output_scores(res,down_value,ideal_ffn,reference):
    arrays=[np.asarray(v,np.float64) for v in (res,down_value,ideal_ffn,reference)]
    if arrays[0].ndim!=1 or not arrays[0].size or any(v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays):
        raise ValueError('complete finite output vectors required')
    y=np.add(arrays[0].astype(np.float32),arrays[1].astype(np.float32),dtype=np.float32)
    return dict(conditioned=proj.metric(arrays[1],arrays[2]),independent=proj.metric(y,arrays[3]))


def attention(lib,values,qn,kn,freq,chunks):
    values=np.asarray(values,np.float32);qn=np.asarray(qn,np.float32);kn=np.asarray(kn,np.float32);freq=np.asarray(freq,np.float64)
    if (values.shape!=(ROWS,14336) or qn.shape!=(256,) or kn.shape!=(256,) or freq.shape!=(32,)
        or not all(np.isfinite(v).all() for v in (values,qn,kn,freq))
        or not chunks or any(type(n)!=int or n<1 or n>T for n in chunks) or sum(chunks)!=ROWS):
        raise ValueError('invalid causal attention geometry/constants')
    kv=np.zeros((ROWS,4096),np.uint16);start=0;outputs=[]
    for count in chunks:
        args=[]
        for v in split(values).values():
            padded=np.full((T,v.shape[1]),np.nan,np.float32);padded[:count]=v[start:start+count];args.append(padded)
        y=np.full((T,6144),np.nan,np.float32)
        geom=np.array([T,count,24,4,256,64,start],np.uintp)
        if lib.prefill_attention(geom,*args,np.ascontiguousarray(qn),np.ascontiguousarray(kn),np.ascontiguousarray(freq),kv,y):
            raise RuntimeError('native attention failed')
        if not np.isfinite(y).all() or np.any(y[count:]):raise ValueError('attention padding/nonfinite output')
        outputs.append(y[:count].copy());start+=count
    return np.concatenate(outputs),kv


def cache_scores(bits,planes):
    bits,planes=np.asarray(bits),np.asarray(planes,np.float64)
    if bits.dtype!=np.uint16 or bits.shape!=(ROWS,4096) or planes.shape!=bits.shape or not np.isfinite(planes).all():
        raise ValueError('complete paired cache planes required')
    got=host.kv_values(bits);kh,kl,vh,vl=np.split(planes,4,axis=1)
    ref=np.concatenate((kh+kl,vh+vl),axis=1)
    return [host.metric(got[t,h*256:(h+1)*256],ref[t,h*256:(h+1)*256]) for t in range(ROWS) for h in range(8)]


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,model_acceptance=False,artifacts={},weights={})
    try:
        attribution=a.attribution.resolve();old=json.loads((attribution/'report.json').read_text())
        if not old['complete'] or old['model_acceptance']:raise ValueError('invalid attribution')
        verify(old['sources']);verify(old['inputs']);verify({str(attribution/p):h for p,h in old['artifacts'].items()})
        capture=a.capture.resolve();audit,meta,refs,evidence=attr.validate(capture,a.audit.resolve())
        spec=base.spec_from_model_dir(Path(meta['model']))
        if meta['layers']!=64 or len(meta['ids'])!=ROWS or spec.norm_eps!=1e-6:raise ValueError('unsupported fixture')
        bundle=proj.load_bundle(Path(meta['kernels']),proj.manifest(spec,max_ctx=meta['max_ctx']))
        if any(bundle[k]!=meta[k] for k in ('build_key','spec_hash')):raise ValueError('bundle changed')
        cases={c['tag'].removeprefix('L3_'):c for c in proj.cases(bundle) if c['tag'].startswith('L3_')}
        model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=5120;model.native_q8=lambda name:False
        retained=next(c for c in old['layers'] if c['layer']==63)
        arrays={}
        with np.load(attribution/'L63_P1.npz') as z:
            for key in ('reference_res','reference_output','native_output'):arrays[key]=z[key].copy()
            arrays['native_output']=native_vector(arrays['native_output'])
            arrays['entry_reference']=np.concatenate([z[k+'_exact'] for k,_,_ in ROLES],axis=1)
        for kind,c in cases.items():
            if (c['n'],c['k'])!=SHAPES[kind]:raise ValueError('wrong projection shape')
            pool=np.zeros(c['n']*c['k']*5//8,np.uint8);offset=0;r['weights'][kind]=[]
            for role,op in zip(c['roles'],c['ops']):
                w,e=attr.packed_weights(model,63,op);r['weights'][kind].append(e)
                expected=retained['attention_weights'][role if kind=='entry' else 'out'] if kind in ('entry','out') else retained['down_weight'] if kind=='down' else e
                if e!=expected:raise ValueError('weight identity differs from attribution')
                del w
                proj.apply_op(dict(op,dst=offset),model,63,pool);offset+=op['nch']*5120
            if offset!=pool.size:raise ValueError('incomplete packed weights')
            path=out/(kind+'-weights.bin');path.write_bytes(pool.tobytes());r['artifacts'][str(path)]=digest(path)
        for key,width,n in [('xn',5120,ROWS),('q',6144,ROWS),('k',1024,ROWS),('v',1024,ROWS),('gate',6144,ROWS),('og',6144,ROWS),
                            ('proj',5120,1),('res',5120,1),('norm',5120,1),('upgate',34816,1),('h',17408,1),('down',5120,1)]:
            arrays[key]=base.read(capture/f'hybrid_L63_P1_{key}.bin',n*width).astype(np.float32).reshape(n,width)
        arrays['kv']=np.fromfile(capture/'hybrid_L63_P1_kv_after.bin','<u2').reshape(ROWS,4096)
        arrays['incoming']=attr.capture.trace.read_trace(capture/'hybrid_res62.bin',21,5120)[:ROWS].astype(np.float32)
        pre='model.layers.63.'
        for key,name in [('qn','self_attn.q_norm.weight'),('kn','self_attn.k_norm.weight'),('norm_w','post_attention_layernorm.weight'),('entry_norm_w','input_layernorm.weight')]:
            arrays[key]=model.bf16(pre+name).astype(np.float32)
        arrays['freq']=np.asarray(spec.rope_inv_freq(),np.float64)
        np.savez(out/'fixture.npz',**arrays);r['artifacts'][str(out/'fixture.npz')]=digest(out/'fixture.npz')
        kernels={kind:str(Path(meta['kernels'])/c['kernel']) for kind,c in cases.items()}
        for kernel in kernels.values():
            for name in ('final.xclbin','insts.bin'):evidence[str(Path(kernel)/name)]=digest(Path(kernel)/name)
        evidence[str(attribution/'report.json')]=digest(attribution/'report.json')
        evidence[str(Path(__file__).resolve())]=digest(__file__)
        r.update(complete=True,inputs=evidence,kernels=kernels,model=meta['model'],
                 scope='L63/P1 composed suffix on actual incoming 20-token prefix')
    except Exception as e:r['error']=str(e);raise
    finally:(out/'preparation.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    return 0


def evaluate(a,execute):
    out=a.out.resolve()
    if execute:out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,control_valid=False,model_acceptance=False,dispatches=0,evidence={},artifacts={},cases=[])
    def pin(path):r['evidence'][str(path.resolve())]=digest(path)
    def save(path,value):
        data=np.ascontiguousarray(value).tobytes()
        if execute:path.write_bytes(data)
        elif path.read_bytes()!=data:raise ValueError('recomputed artifact differs: '+str(path))
        r['artifacts'][str(path)]=digest(path)
    lock=None
    try:
        prepared=a.prepared.resolve();prep=json.loads((prepared/'preparation.json').read_text())
        if not prep['complete']:raise ValueError('incomplete preparation')
        verify(prep['inputs']);verify(prep['artifacts']);pin(prepared/'preparation.json')
        roots=dict(entry=a.build_root.resolve(),out=a.out_build.resolve(),up_gate=a.ffn_build.resolve(),down=a.down_build.resolve())
        for kind,path in roots.items():
            for mode in ('control','low'):
                b=paired.verify_build(path/mode,mode);n,k=SHAPES[kind]
                if b.get('shape',dict(n=n,k=k,t=T))!=dict(n=n,k=k,t=T) or any(b['environment'].get(key)!=v for key,v in dict(GQP_N=str(n),GQP_K=str(k),GQP_T=str(T),GQP_FMT='q4_1',GQP_BREUSE='0').items()):
                    raise ValueError('wrong '+kind+' kernel geometry')
                pin(path/mode/'build.json')
        hb=a.harness_build.resolve();down.harness(hb);pin(hb/'build-evidence.json')
        bridgepath=a.bridge.resolve();b=json.loads((bridgepath/'build.json').read_text())
        if not b['complete']:raise ValueError('incomplete attention bridge')
        verify(b['sources']);verify(b['dependencies']);verify({str(bridgepath/'attention.so'):b['binary_sha256']});pin(bridgepath/'build.json')
        lib=host.native_library(bridgepath/'attention.so');ffn=bridge.ffn.Host(bridgepath/'ffn');pin(bridgepath/'ffn/build.json')
        with np.load(prepared/'fixture.npz') as z:f={k:z[k].copy() for k in z.files}
        captured=np.concatenate([f[k] for k,_,_ in ROLES],axis=1)
        for t in range(ROWS):prefix.identity(ffn.norm(f['incoming'][t],f['entry_norm_w'],1e-6),f['xn'][t],'native entry norm')
        def host_call(x):
            og,kv=attention(lib,x,f['qn'],f['kn'],f['freq'],[ROWS])
            seq,skv=attention(lib,x,f['qn'],f['kn'],f['freq'],[1]*ROWS)
            prefix.identity(og,seq,'attention partition');prefix.identity(kv,skv,'KV partition')
            return og,kv
        og,kv=host_call(captured)
        prefix.identity(og,f['og'],'native attention');prefix.identity(kv,f['kv'],'native paired KV')
        prefix.identity(ffn.norm(f['res'][0],f['norm_w'],1e-6),f['norm'][0],'native norm')
        prefix.identity(ffn.activation(f['upgate'][0]),f['h'][0],'native activation')
        model=base.Q4NX(Path(prep['model'])/'model.q4nx');model.hidden=5120;model.native_q8=lambda name:False
        spec=base.spec_from_model_dir(Path(prep['model']))
        def ideal(kind,x):
            n,k=SHAPES[kind];w=proj.dequant_pool(np.fromfile(prepared/(kind+'-weights.bin'),np.uint8),n,k,rs=2)
            y=attr.rounding.predict(x,w)['exact'];del w;return y
        prefix.identity(ideal('entry',f['xn']),f['entry_reference'],'primary entry reference')
        if execute:
            lock=(ROOT/'Models/qwen38-27b/.resident-validation.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);down.runner.device_available()
        def dispatch(kind,label,mode,x,offset):
            n,k=SHAPES[kind];kernel=Path(prep['kernels'][kind]) if mode=='production' else roots[kind]/mode
            products=[];weights=prepared/(kind+'-weights.bin')
            for part,values in zip(('high','low'),paired.control_tool.split_inputs(x)):
                tag=f'{offset}-{kind}-{label}-{mode}-{part}';xp=out/(tag+'-x.bin');yp=out/(tag+'-y.bin');cfg=out/(tag+'.cfg')
                save(xp,proj.tile_input(prefix.place(values,offset)))
                text='\n'.join(['device',f'xclbin G {kernel}/final.xclbin',f'kernelx k G {kernel}/insts.bin',
                     f'buf w {weights.stat().st_size} {weights}',f'buf x {xp.stat().st_size} {xp}',f'buf y {n*T*4}',
                     'run k w x y',f'dump y {yp} {n*T*4}',''])
                if execute:
                    down.runner.device_available();cfg.write_text(text)
                    with cfg.with_suffix('.log').open('w') as log:
                        subprocess.run([str(hb/'build/run_kernel'),str(cfg)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120,
                                       env=os.environ|{'HARNESS_TIMEOUT_MS':'30000'})
                elif cfg.read_text()!=text:raise ValueError('dispatch recipe changed')
                pin(cfg);pin(cfg.with_suffix('.log'));r['artifacts'][str(yp)]=digest(yp);r['dispatches']+=1
                products.append(prefix.extract(base.read(yp,n*T).astype(np.float32).reshape(n,T),offset,len(x)))
            return products
        def projection(kind,label,x,offset,weighted,native=None):
            high=dispatch(kind,label,'control',x,offset)
            if native is not None or (kind=='out' and label=='baseline'):
                prod=dispatch(kind,label,'production',x,offset);paired.require_control(*high,*prod)
            y=paired.combine(*high,*dispatch(kind,label,'low',x,offset)) if weighted else np.add(*high,dtype=np.float32)
            if native is not None:prefix.identity(y,native,'native '+kind)
            return y
        candidates=[]
        for offset in (0,128):
            control_entry=projection('entry','baseline',f['xn'],offset,False,captured)
            candidate_entry=paired.combine(control_entry,np.zeros_like(control_entry),*dispatch('entry','baseline','low',f['xn'],offset))
            e=entry_scores(control_entry,candidate_entry,f['entry_reference'])
            cog,ckv=host_call(candidate_entry)
            independent_og,independent_kv=attr.capture.attention_reference({k:v.astype(np.float64) for k,v in split(candidate_entry).items()},f['qn'].astype(np.float64),f['kn'].astype(np.float64),f['freq'],24,4,256,1e-6)
            host_checks=[host.metric(cog[t,h*256:(h+1)*256],independent_og[t,h*256:(h+1)*256]) for t in range(ROWS) for h in range(24)]
            host_checks += cache_scores(ckv,independent_kv)
            baseline_out=projection('out','baseline',og,offset,False)
            prefix.identity(baseline_out[P],f['proj'][0],'native out projection')
            candidate_out=projection('out','candidate',cog,offset,True)
            out_ref=ideal('out',cog);out_checks=[proj.metric(y,z) for y,z in zip(candidate_out,out_ref)]
            baseline_res=np.add(f['incoming'][P],baseline_out[P],dtype=np.float32)
            prefix.identity(baseline_res,f['res'][0],'native pre-FFN residual')
            res=np.add(f['incoming'][P],candidate_out[P],dtype=np.float32)
            norm=ffn.norm(res,f['norm_w'],1e-6)[None,:]
            ug0=projection('up_gate','baseline',f['norm'],offset+P,True,f['upgate'])
            h0=ffn.activation(ug0[0])[None,:];prefix.identity(h0,f['h'],'native activation')
            ug=projection('up_gate','candidate',norm,offset+P,True);ugref=ideal('up_gate',norm)
            ug_checks=[proj.metric(ug[0,i:i+17408],ugref[0,i:i+17408]) for i in (0,17408)]
            h=ffn.activation(ug[0])[None,:]
            d0=projection('down','baseline',h0,offset+P,False,f['down'])
            prefix.identity(np.add(baseline_res,d0[0],dtype=np.float32),f['native_output'],'native full layer output')
            d=projection('down','candidate',h,offset+P,True);dref=ideal('down',h)
            output=np.add(res,d[0],dtype=np.float32)
            ideal_ffn=base.oracle.ffn(model,spec,63,res)
            scores=output_scores(res,d[0],ideal_ffn,f['reference_output'])
            case=dict(offset=offset,entry=e,host_checks=host_checks,out_checks=out_checks,upgate_checks=ug_checks,
                down=proj.metric(d,dref),pre_ffn=proj.metric(res.astype(np.float64),f['reference_res'].astype(np.float64)),
                control_output=proj.metric(f['native_output'].astype(np.float64),f['reference_output'].astype(np.float64)),**scores)
            r['cases'].append(case)
            row=dict(entry=candidate_entry,og=cog,kv=ckv,out=candidate_out,res=res,norm=norm,upgate=ug,h=h,down=d,output=output)
            for key,v in row.items():save(out/f'candidate-{offset}-{key}.bin',v)
            candidates.append(row)
            print(json.dumps({k:v for k,v in case.items() if k not in ('entry','host_checks','out_checks','upgate_checks')}),flush=True)
        for key in candidates[0]:prefix.identity(candidates[0][key],candidates[1][key],'candidate slot '+key)
        projection_passed=all(c['entry']['passed'] and c['entry']['improved'] and c['down']['passed'] and all(x['passed'] for x in c['out_checks']+c['upgate_checks']) for c in r['cases'])
        r.update(complete=True,control_valid=True,slot_identity=True,projection_passed=projection_passed,
                 host_passed=all(x['passed'] for c in r['cases'] for x in c['host_checks']),
                 conditioned_passed=all(c['conditioned']['passed'] for c in r['cases']),
                 residual_passed=all(c['independent']['passed'] for c in r['cases']),
                 scope='isolated L63/P1 suffix; incoming prefix retained, not fresh model propagation')
        for p in (Path(__file__),Path(bridge.__file__),Path(down.__file__),Path(attr.__file__),Path(prefix.__file__),Path(paired.__file__)):pin(p)
        verify(r['evidence']);verify(r['artifacts']);verify(prep['inputs']);verify(prep['artifacts'])
        if not execute:
            saved=json.loads((out/'report.json').read_text())
            if r!=saved:raise ValueError('recomputed report differs')
    except Exception as error:r.update(complete=False,control_valid=False,error=str(error));raise
    finally:
        if lock is not None:lock.close()
        (out/('report.json' if execute else 'audit.json')).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    return 0 if all(r[k] for k in ('projection_passed','host_passed','conditioned_passed','residual_passed')) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for key in ('out','attribution','capture','audit'):a.add_argument('--'+key,type=Path,required=True)
    for command in ('run','audit'):
        a=sub.add_parser(command)
        for key in ('out','prepared','build-root','out-build','ffn-build','down-build','harness-build','bridge'):a.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();raise SystemExit(prepare(args) if args.command=='prepare' else evaluate(args,args.command=='run'))
