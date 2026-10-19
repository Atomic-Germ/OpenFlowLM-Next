"""Four- or 64-layer causal prefill and NPU continuation against a CPU oracle."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'open_kernels'))
sys.path.insert(0,str(ROOT/'open_kernels/model'))
from q4nx import Q4NX, dq_chunks_q8
from recipes.load import spec_from_model_dir
from recipes.qwen35 import layout
from recipes.manifest import manifest
from decode_bundle import load_bundle
import replica_qwen35 as oracle
from replica_prompt import prompt_ids
from compare_decode import LOGIT_CORRELATION_MIN, RESIDUAL_MAXREL


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path,count,sha=None):
    b=Path(path).read_bytes()
    if len(b)!=count*4 or (sha is not None and hashlib.sha256(b).hexdigest()!=sha):
        raise ValueError(f'incomplete or changed capture: {path}')
    a=np.frombuffer(b,'<f4').astype(np.float64)
    if not np.isfinite(a).all():raise ValueError(f'non-finite capture: {path}')
    return a


def expected_dumps(prefix,tokens):
    if type(tokens)!=int or tokens<1:raise ValueError('positive token count required')
    return [f'{prefix}_p{t}.bin' for t in range(tokens)]+[f'{prefix}_t0.bin',f'{prefix}_t1.bin']


def logit_score(got,ref,real_vocab):
    got,ref=[np.asarray(v,dtype=np.float64) for v in (got,ref)]
    if (got.ndim!=1 or got.shape!=ref.shape or not 1<real_vocab<=got.size
            or not all(np.isfinite(v).all() and np.ptp(v)>0 for v in (got,ref))):
        raise ValueError('complete nonconstant finite logits required')
    corr=float(np.corrcoef(got,ref)[0,1])
    g,r=int(got[:real_vocab].argmax()),int(ref[:real_vocab].argmax())
    return dict(passed=corr>LOGIT_CORRELATION_MIN and g==r,correlation=corr,argmax=g,reference_argmax=r)


def check_tokens(log,expected):
    got=[int(v) for v in re.findall(r'^token (\d+)\s*$',log,re.M)]
    if got!=expected:raise ValueError(f'continuation tokens differ: {got} != {expected}')


def final_residual(data,offsets,hidden):
    if len(data)!=offsets.AA_BYTES:raise ValueError('incomplete final act buffer')
    values=[]
    for field in ('AA_RES','AA_OUT2','AA_OUT2B'):
        off=getattr(offsets,field)
        if off<0 or off%4 or off+hidden*4>len(data):raise ValueError('invalid residual region')
        values.append(np.frombuffer(data,'<f4',count=hidden,offset=off).astype(np.float64))
    result=sum(values)
    if not np.isfinite(result).all():raise ValueError('non-finite residual')
    return result


def verify(hashes):
    for name,h in hashes.items():
        if digest(name)!=h:raise ValueError(f'provenance mismatch: {name}')


def head_batch(model,hn,vocab):
    """Stream 256 vocabulary rows at a time, sharing Q8 unpacking across tokens."""
    hn=np.asarray(hn,dtype=np.float64);hidden=model.hidden
    if (hn.ndim!=2 or not hn.shape[0] or hn.shape[1]!=hidden or hidden%256
            or vocab<=0 or vocab%32 or not np.isfinite(hn).all()
            or model.chunk_bytes_of('lm_head.weight')!=8704):
        raise ValueError('invalid Q8 head geometry/input')
    raw=model.raw('lm_head.weight')
    if len(raw)!=vocab*hidden//8192*8704:raise ValueError('incomplete Q8 head')
    chunks=np.frombuffer(raw,np.uint8).reshape(-1,8704);ncol=hidden//256
    out=np.empty((hn.shape[0],vocab),np.float32)
    for row in range(0,vocab,256):
        rows=min(256,vocab-row);lo=row//32*ncol;hi=(row+rows)//32*ncol
        w=dq_chunks_q8(chunks[lo:hi]).reshape(rows//32,ncol,32,256)
        w=w.transpose(0,2,1,3).reshape(rows,hidden).astype(np.float64)
        out[:,row:row+rows]=hn@w.T
    return out


def tail_batch(model,spec,xs):
    hn=(oracle.rms(np.asarray(xs),spec.norm_eps)*model.bf16('model.norm.weight')).astype(np.float32)
    return head_batch(model,hn,spec.vocab)


def prompt_cases(ids,boundaries):
    lengths=(255,256,257) if boundaries else (len(ids),)
    if not ids or len(ids)<max(lengths):raise ValueError('insufficient tokens for requested prefixes')
    return {n:ids[:n] for n in lengths}


def causal_prefixes(xs,state,lengths,decode):
    """Preserve independent causal checkpoints even when decode mutates state."""
    if (not lengths or len(set(lengths))!=len(lengths)
            or any(type(n)!=int or not 1<=n<=len(xs) for n in lengths)):
        raise ValueError('invalid causal checkpoints')
    rows=[];states={}
    for pos,x in enumerate(xs[:max(lengths)]):
        y,state=decode(x,state,pos)
        rows.append(np.array(y,dtype=np.float32,copy=True))
        if pos+1 in lengths:states[pos+1]=tuple(np.array(v,copy=True) for v in state)
    return rows,states


def checked_layers(layers):
    if type(layers)!=int or layers not in (4,64):raise ValueError('layers must be 4 or 64')
    return layers


def check_reference_coverage(meta):
    layers=checked_layers(meta['layers']);n=len(meta['ids'])
    required={f'ref_{part}res{i}.bin' for part in ('','cont_') for i in range(layers)}
    required.update(f'ref_p{i}.bin' for i in range(n));required.add('ref_cont_logits.bin')
    if n<1 or set(meta['references'])!=required:raise ValueError('incomplete reference coverage')


def memory_preflight(layers,available=None):
    """Full resident validation needs measured headroom beyond the weight file."""
    checked_layers(layers)
    if available is None:
        match=re.search(r'^MemAvailable:\s+(\d+)\s+kB$',Path('/proc/meminfo').read_text(),re.M)
        if match is None:raise RuntimeError('cannot determine available RAM')
        available=int(match[1])*1024
    if layers==64 and available<35*2**30:
        raise RuntimeError(f'64-layer resident run requires at least 35 GiB available RAM; available {available/2**30:.1f} GiB')
    return available


def native_command(cli,out,meta,offsets,route,extra):
    layers=checked_layers(meta['layers'])
    return [str(cli.resolve()),'--model',meta['model'],'--kernels',meta['kernels'],
            '--ids-file',str(out/'ids.txt'),'--layers',str(layers),'--max-ctx',str(meta['max_ctx']),
            '--max-tokens','2','--decode-route','npu','--pmode','none','--prefill-logits',
            '--dump-logits',str(out/route),'--dump-act',f'{layers-1}:0:{offsets.AA_BYTES}:{out}/{route}-act.bin',*extra]


def prepare(a):
    from tokenizers import Tokenizer
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ('reference.json','report.json'):(out/name).unlink(missing_ok=True)
    boundaries=getattr(a,'boundaries',False)
    if boundaries:
        for n in (255,256,257):
            for name in ('reference.json','report.json'):(out/str(n)/name).unlink(missing_ok=True)
    spec=spec_from_model_dir(a.model)
    layers=checked_layers(getattr(a,'layers',4))
    m0=json.loads((a.kernels/'manifest.json').read_text())
    m=load_bundle(a.kernels,manifest(spec,max_ctx=m0['max_ctx_default']))
    if (spec.family!='qwen35' or spec.hidden!=5120 or spec.num_layers!=64
            or list(spec.layer_types)!=(['linear_attention']*3+['full_attention'])*16):
        raise ValueError('requires the 27B linear/full layer geometry')
    source={};original=json.loads(a.original.read_text())
    path=(a.model/'model.q4nx').resolve();h=digest(path)
    prior={str((ROOT/Path(p)).resolve()):v for p,v in original.items()}
    if prior.get(str(path))!=h:raise ValueError('model differs from original provenance')
    source[str(path)]=h
    for p in [a.original,a.model/'config.json',a.model/'tokenizer.json',a.kernels/'manifest.json',a.kernels/'spec.json',a.kernels/'toolchain.json']:
        source[str(p.resolve())]=digest(p)
    tk=Tokenizer.from_file(str(a.model/'tokenizer.json'))
    ids=prompt_ids(tk,a.message,False)
    cases=prompt_cases(ids,boundaries);ids=cases[max(cases)]
    if any(type(t)!=int or not 0<=t<spec.real_vocab for t in ids) or len(ids)+1>m['max_ctx_default']:
        raise ValueError('invalid prompt ids/context')
    dirs={n:out/str(n) if boundaries else out for n in cases}
    for n,d in dirs.items():
        d.mkdir(parents=True,exist_ok=True)
        (d/'ids.txt').write_text(','.join(map(str,cases[n]))+'\n')
    print(f'prompt: {len(ids)} tokens; {layers} layers; one continuation step',flush=True)
    model=Q4NX(path);model.hidden=spec.hidden;model.native_q8=lambda name:False
    plain=model.matmul_w;cache={}
    def cached(name,o,i):
        if name not in cache:cache[name]=plain(name,o,i)
        return cache[name]
    model.matmul_w=cached
    xs=[model.embed(t,spec.hidden).astype(np.float32) for t in ids]
    states=[];refs={n:{} for n in cases}
    def save(n,name,value):
        p=dirs[n]/name;np.asarray(value,dtype='<f4').tofile(p);refs[n][name]=digest(p)
    for layer in range(layers):
        cache.clear()
        conv=np.zeros((spec.conv_kernel-1,spec.lin_qkv_dim))
        state=np.zeros((spec.lin_value_heads,spec.lin_value_dim,spec.lin_value_dim))
        k=np.zeros((0,spec.num_kv_heads,spec.head_dim));v=k.copy()
        def decode(x,s,pos):
            y,*s=oracle.layer_decode(model,spec,layer,x,*s,pos)
            return y,s
        xs,snapshots=causal_prefixes(xs,(conv,state,k,v),tuple(cases),decode)
        states.append(snapshots)
        for n in cases:save(n,f'ref_res{layer}.bin',np.stack(xs[:n]))
        print(f'CPU prompt layer {layer} complete',flush=True)
    cache.clear()
    logits=tail_batch(model,spec,xs)
    for n in cases:
        for pos,y in enumerate(logits[:n]):save(n,f'ref_p{pos}.bin',y)
    print(f'CPU logits: {len(ids)} positions complete',flush=True)
    continuation={n:int(logits[n-1][:spec.real_vocab].argmax()) for n in cases}
    continuation_x={n:model.embed(t,spec.hidden).astype(np.float32) for n,t in continuation.items()}
    for layer in range(layers):
        cache.clear()
        for n in cases:
            x,*state=oracle.layer_decode(model,spec,layer,continuation_x[n],*states[layer][n],n)
            continuation_x[n]=np.asarray(x,np.float32);save(n,f'ref_cont_res{layer}.bin',continuation_x[n])
        print(f'CPU continuation layer {layer} complete',flush=True)
    cache.clear();last=tail_batch(model,spec,list(continuation_x.values()))
    sources=[Path(__file__),ROOT/'open_kernels/model/replica_qwen35.py',ROOT/'open_kernels/model/replica_prompt.py',ROOT/'open_kernels/model/q4nx.py',ROOT/'open_kernels/model/decode_bundle.py',ROOT/'open_kernels/model/compare_decode.py']
    sources+=list((ROOT/'open_kernels/recipes').glob('*.py'))
    for p in sources:source[str(p.resolve())]=digest(p)
    for (n,d),y in zip(dirs.items(),last):
        save(n,'ref_cont_logits.bin',y)
        expected=[continuation[n],int(y[:spec.real_vocab].argmax())]
        meta=dict(version=1,layers=layers,ids=cases[n],message=a.message,expected_tokens=expected,
                  prompt_kind='chat_prefix' if boundaries else 'complete_chat',
                  vocab=spec.vocab,real_vocab=spec.real_vocab,hidden=spec.hidden,
                  build_key=m['build_key'],spec_hash=m['spec_hash'],max_ctx=m['max_ctx_default'],
                  sources=source,references=refs[n],ids_sha256=digest(d/'ids.txt'),
                  model=str(a.model.resolve()),kernels=str(a.kernels.resolve()))
        check_reference_coverage(meta)
        (d/'reference.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(f'CPU expected continuation at {n}: {expected}',flush=True)


def run(a):
    out=a.out.resolve();report_path=out/'report.json';report_path.unlink(missing_ok=True)
    report=dict(passed=False,routes=[])
    try:
        meta=json.loads((out/'reference.json').read_text());check_reference_coverage(meta)
        layers=checked_layers(meta['layers'])
        report.update(layers=layers,last_residual_layer=layers-1,
                      scope=f'{layers}-layer causal prompt, all prompt logits and one NPU continuation')
        report['available_memory_bytes']=memory_preflight(layers)
        verify(meta['sources'])
        verify({str(out/p):h for p,h in meta['references'].items()})
        if digest(out/'ids.txt')!=meta['ids_sha256']:raise ValueError('prompt ids changed')
        spec=spec_from_model_dir(Path(meta['model']));offsets=layout(spec,max_ctx=meta['max_ctx'])
        load_bundle(Path(meta['kernels']),manifest(spec,max_ctx=meta['max_ctx']))
        report.update(reference_sha256=digest(out/'reference.json'),cli_sha256=digest(a.cli),
                      prompt_tokens=len(meta['ids']))
        # Isolate explicit route selection from ambient experiment flags.
        env={k:v for k,v in os.environ.items() if not k.startswith('OFLM_OPEN_')}
        env.update(OMP_NUM_THREADS='2',OMP_WAIT_POLICY='PASSIVE',OFLM_OPEN_ATTN_BLOCK='0',
                   OFLM_OPEN_LAYER_MAJOR='1',OFLM_OPEN_STAGE_MAJOR='0')
        captures={};n=len(meta['ids'])
        for route,extra in [('sequential',[]),('hybrid',['--gemm-block']),('block_major',['--gemm-block','--block-major'])]:
            memory_preflight(layers)
            names=expected_dumps(route,n)
            for name in names+[route+'-act.bin']:(out/name).unlink(missing_ok=True)
            cmd=native_command(a.cli,out,meta,offsets,route,extra)
            log=out/(route+'.log')
            print('running',route,flush=True)
            with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=1800,check=True)
            arrays={name:read(out/name,meta['vocab']) for name in names}
            rows=[]
            for pos in range(n):
                ref=read(out/f'ref_p{pos}.bin',meta['vocab'],meta['references'][f'ref_p{pos}.bin'])
                rows.append(dict(position=pos,**logit_score(arrays[f'{route}_p{pos}.bin'],ref,meta['real_vocab'])))
            ref=read(out/'ref_cont_logits.bin',meta['vocab'],meta['references']['ref_cont_logits.bin'])
            rows.append(dict(position=n,**logit_score(arrays[f'{route}_t1.bin'],ref,meta['real_vocab'])))
            tokens_ok=True;token_error=None
            try:check_tokens(log.read_text(),meta['expected_tokens'])
            except ValueError as e:tokens_ok=False;token_error=str(e)
            alias=(out/f'{route}_t0.bin').read_bytes()==(out/f'{route}_p{n-1}.bin').read_bytes()
            residual=final_residual((out/f'{route}-act.bin').read_bytes(),offsets,meta['hidden'])
            residual_name=f'ref_cont_res{layers-1}.bin'
            ref=read(out/residual_name,meta['hidden'],meta['references'][residual_name])
            err=float(np.max(np.abs(residual-ref))/max(float(np.max(np.abs(ref))),1e-30))
            hashes={name:digest(out/name) for name in names+[route+'-act.bin']};captures[route]=hashes
            result=dict(route=route,passed=tokens_ok and alias and err<RESIDUAL_MAXREL and all(r['passed'] for r in rows),
                        command=cmd,log_sha256=digest(log),tokens_match=tokens_ok,token_error=token_error,
                        last_prompt_alias=alias,continuation_last_residual_maxrel=err,logits=rows,outputs=hashes)
            report['routes'].append(result)
            print(route,'passed',result['passed'],'min corr',min(r['correlation'] for r in rows),'residual',err,flush=True)
        pairs={name:captures['hybrid'][name]==captures['block_major'][name.replace('hybrid','block_major',1)]
               for name in expected_dumps('hybrid',n)}
        report['hybrid_schedule_identity']=pairs
        report['passed']=all(c['passed'] for c in report['routes']) and all(pairs.values())
    except Exception as e:
        report['error']=str(e)
        raise
    finally:report_path.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for key in ('out','model','kernels','original'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--message',default='What is 2 + 2?')
    p.add_argument('--layers',type=int,choices=(4,64),default=4)
    p.add_argument('--boundaries',action='store_true',help='prepare shared causal prefixes of 255/256/257 tokens; message must be long enough')
    p=sub.add_parser('run');p.add_argument('--out',type=Path,required=True);p.add_argument('--cli',type=Path,required=True)
    a=ap.parse_args();return prepare(a) if a.command=='prepare' else run(a)


if __name__=='__main__':raise SystemExit(main())
