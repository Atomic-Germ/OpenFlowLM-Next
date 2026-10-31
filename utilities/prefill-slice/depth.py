"""Locate hybrid prefill drift by comparing retained causal references at several depths."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

_spec=importlib.util.spec_from_file_location('_prefill_depth_base',Path(__file__).with_name('check.py'))
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)


def depths(text):
    values=[int(v) for v in text.split(',')]
    if not values or values!=sorted(set(values)) or any(n<4 or n>64 or n%4 for n in values):
        raise ValueError('unique increasing depths divisible by four, from 4 through 64, required')
    return values


def score_rows(got,ref,real_vocab):
    got,ref=map(np.asarray,(got,ref))
    if got.ndim!=2 or got.shape!=ref.shape or not len(ref):raise ValueError('complete position coverage required')
    rows=[dict(position=i,**base.logit_score(g,r,real_vocab)) for i,(g,r) in enumerate(zip(got,ref))]
    return dict(passed=all(r['passed'] for r in rows),logits=rows,
                min_correlation=min(r['correlation'] for r in rows))


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ('reference.json','report.json'):(out/name).unlink(missing_ok=True)
    fixture=a.fixture.resolve();meta=json.loads((fixture/'reference.json').read_text())
    base.check_reference_coverage(meta)
    if meta['layers']!=64:raise ValueError('a complete 64-layer CPU reference is required')
    base.verify(meta['sources']);base.verify({str(fixture/p):h for p,h in meta['references'].items()})
    if base.digest(fixture/'ids.txt')!=meta['ids_sha256']:raise ValueError('prompt changed')
    spec=base.spec_from_model_dir(Path(meta['model']))
    base.load_bundle(Path(meta['kernels']),base.manifest(spec,max_ctx=meta['max_ctx']))
    selected=depths(a.depths);n=len(meta['ids'])
    xs=np.concatenate([base.read(fixture/f'ref_res{d-1}.bin',n*spec.hidden).reshape(n,spec.hidden)
                       for d in selected])
    model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=spec.hidden
    logits=base.tail_batch(model,spec,xs).reshape(len(selected),n,spec.vocab)
    refs={}
    for d,rows in zip(selected,logits):
        p=out/f'ref_depth{d}.bin';np.asarray(rows,dtype='<f4').tofile(p);refs[p.name]=base.digest(p)
    (out/'ids.txt').write_bytes((fixture/'ids.txt').read_bytes())
    sources=dict(meta['sources']);sources[str(Path(__file__).resolve())]=base.digest(__file__)
    sources[str(fixture/'reference.json')]=base.digest(fixture/'reference.json')
    for d in selected:
        p=fixture/f'ref_res{d-1}.bin';sources[str(p)]=meta['references'][p.name]
    result=dict(depths=selected,ids=meta['ids'],ids_sha256=meta['ids_sha256'],
                model=meta['model'],kernels=meta['kernels'],max_ctx=meta['max_ctx'],
                vocab=spec.vocab,real_vocab=spec.real_vocab,sources=sources,references=refs)
    (out/'reference.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Prepared independent tail references for depths',selected,flush=True)


def run(a):
    out=a.out.resolve();path=out/'report.json';path.unlink(missing_ok=True)
    report=dict(passed=False,cases=[],scope='hybrid depth diagnostic; no continuation or per-layer boundary acceptance')
    try:
        meta=json.loads((out/'reference.json').read_text())
        selected=depths(','.join(map(str,meta['depths'])));n=len(meta['ids'])
        if set(meta['references'])!={f'ref_depth{d}.bin' for d in selected}:raise ValueError('incomplete depth references')
        base.memory_preflight(64);base.verify(meta['sources'])
        base.verify({str(out/p):h for p,h in meta['references'].items()})
        if base.digest(out/'ids.txt')!=meta['ids_sha256']:raise ValueError('prompt changed')
        spec=base.spec_from_model_dir(Path(meta['model']))
        base.load_bundle(Path(meta['kernels']),base.manifest(spec,max_ctx=meta['max_ctx']))
        report.update(reference_sha256=base.digest(out/'reference.json'),cli_sha256=base.digest(a.cli))
        env={k:v for k,v in os.environ.items() if not k.startswith('OFLM_OPEN_')}
        env.update(OMP_NUM_THREADS='2',OMP_WAIT_POLICY='PASSIVE',OFLM_OPEN_ATTN_BLOCK='0',
                   OFLM_OPEN_LAYER_MAJOR='1',OFLM_OPEN_STAGE_MAJOR='0')
        for d in selected:
            base.memory_preflight(64)
            prefix=f'depth{d}';names=[f'{prefix}_p{i}.bin' for i in range(n)]+[f'{prefix}_t0.bin']
            for name in names:(out/name).unlink(missing_ok=True)
            cmd=[str(a.cli.resolve()),'--model',meta['model'],'--kernels',meta['kernels'],
                 '--layers',str(d),'--max-ctx',str(meta['max_ctx']),'--ids-file',str(out/'ids.txt'),
                 '--max-tokens','1','--decode-route','npu','--pmode','none','--gemm-block',
                 '--prefill-logits','--dump-logits',str(out/prefix)]
            log=out/f'{prefix}.log'
            print('Running hybrid depth',d,flush=True)
            with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=1800,check=True)
            got=np.stack([base.read(out/name,meta['vocab']) for name in names[:-1]])
            ref=base.read(out/f'ref_depth{d}.bin',n*meta['vocab']).reshape(n,meta['vocab'])
            result=score_rows(got,ref,meta['real_vocab'])
            # A matching final token alone cannot accept failures at earlier positions.
            token_error=None
            try:base.check_tokens(log.read_text(),[int(ref[-1,:meta['real_vocab']].argmax())])
            except ValueError as e:token_error=str(e)
            alias=(out/names[-1]).read_bytes()==(out/names[-2]).read_bytes()
            result.update(depth=d,command=cmd,last_prompt_alias=alias,token_error=token_error,log_sha256=base.digest(log),
                          outputs={name:base.digest(out/name) for name in names})
            result['passed'] &= alias and token_error is None
            report['cases'].append(result)
            print('depth',d,'passed',result['passed'],'min corr',result['min_correlation'],flush=True)
        report['passed']=all(c['passed'] for c in report['cases'])
    except Exception as e:
        report['error']=str(e);raise
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--depths',default='4,8,16,32,48,60')
    p=sub.add_parser('run');p.add_argument('--out',type=Path,required=True);p.add_argument('--cli',type=Path,required=True)
    a=ap.parse_args();return prepare(a) if a.command=='prepare' else run(a)


if __name__=='__main__':raise SystemExit(main())
