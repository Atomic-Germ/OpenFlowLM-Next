"""Validate standard FP32 prefill at 255/256/257 tokens with fresh CLI controls."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


HERE=Path(__file__).resolve().parent
reset=load('_post_reset_boundary',HERE/'post_reset.py')
post,base,digest=reset.post,reset.base,reset.digest
boundary=load('_post_boundary_refs',HERE.parent/'prefill-slice/boundaries.py')
LENGTHS=(255,256,257)


def sources():
    return reset.sources() | {str(Path(p).resolve()):digest(p) for p in (__file__,boundary.__file__)}


def check_reference_binding(meta,short,proof,path,sha,oracle):
    if not proof['passed'] or proof['evidence'].get(str(path))!=sha:
        raise ValueError('reference metadata lacks a bound passing audit')
    if any(meta[k]!=short[k] for k in ('hidden','vocab','real_vocab','model','spec_hash','max_ctx')):
        raise ValueError('boundary reference differs from current short geometry')
    if any(meta['sources'].get(p)!=h for p,h in oracle.items()):
        raise ValueError('boundary model/oracle differs from audited short baseline')


def references(root,short,short_record):
    proof_path=root/'reference-audit.json'
    proof=json.loads(proof_path.read_text())
    inputs={str(proof_path):digest(proof_path)}
    model=Path(short['model'])
    paths=[model/n for n in ('model.q4nx','config.json','tokenizer.json')]
    paths += [base.ROOT/'utilities/prefill-slice/check.py']
    paths += [base.ROOT/'open_kernels/model'/n for n in
              ('replica_qwen35.py','replica_prompt.py','q4nx.py','compare_decode.py')]
    oracle={str(p):short_record['inputs'][str(p)] for p in paths}
    cases={}
    for n in LENGTHS:
        directory=root/str(n);path=directory/'reference.json'
        meta=boundary.validate_reference(directory,64,n)
        check_reference_binding(meta,short,proof,path,digest(path),oracle)
        inputs[str(path)]=digest(path)
        inputs[str(directory/'ids.txt')]=meta['ids_sha256']
        inputs.update({str(directory/name):h for name,h in meta['references'].items()})
        cases[n]=meta
    prefixes={f'{n}/{n+1}':boundary.compare_prefix(root/str(n),cases[n],root/str(n+1),cases[n+1])
              for n in LENGTHS[:-1]}
    if sum(len(m['references']) for m in cases.values())!=proof['references'] or prefixes!=proof['prompt_prefixes']:
        raise ValueError('reference inventory or prefixes differ from audit')
    # The immutable reference metadata describes its original export. Only the
    # runtime view selects the current, separately validated FP32 kernel bundle.
    return {n:m|{'kernels':short['kernels']} for n,m in cases.items()},inputs,prefixes


def context(record):
    if record['sources']!=sources():raise ValueError('boundary validator changed')
    base.verify(record['sources'])
    short,_,offsets,_=reset.baseline(record)
    baseline=json.loads((Path(record['baseline'])/'report.json').read_text())
    if any(record[k]!=baseline[k] for k in ('plain_build','trace_build')):
        raise ValueError('builds differ from fresh audited short run')
    cases,inputs,prefixes=references(Path(record['references']),short,baseline)
    return cases,offsets,inputs,prefixes


def prepare(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve()
    record=dict(complete=False,model_acceptance=False,sources=sources(),baseline=str(baseline),
                baseline_hashes={n:digest(baseline/n) for n in ('report.json','audit.json')},
                references=str(args.references.resolve()))
    try:
        short=json.loads((baseline/'report.json').read_text())
        record.update({k:short[k] for k in ('plain_build','trace_build')})
        _,_,record['inputs'],record['reference_prefixes']=context(record)
        record.update(complete=True,scope='Verified references and current plain/trace builds; no NPU execution')
    except Exception as error:record['error']=str(error);raise
    finally:(out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')


def verify_preparation(directory):
    record=json.loads((directory/'preparation.json').read_text())
    if not record['complete'] or record['model_acceptance'] is not False:
        raise ValueError('incomplete preparation')
    cases,offsets,inputs,prefixes=context(record)
    if inputs!=record['inputs'] or prefixes!=record['reference_prefixes']:
        raise ValueError('prepared inputs changed')
    return record,cases,offsets


def accept(cases,prefixes):
    if set(cases)!={str(n) for n in LENGTHS} or not prefixes:
        raise ValueError('incomplete boundary coverage')
    passed=all(prefixes.values())
    for case in cases.values():
        for key in ('plain','routes'):
            if [r['route'] for r in case[key]]!=list(post.old.runner.ROUTES):
                raise ValueError('incomplete or unordered boundary routes')
        passed &= all(r['passed'] for r in case['plain'])
        passed &= all(r['first']['passed'] and bool(r['identity']) and all(r['identity'].values())
                      for r in case['routes'])
        passed &= bool(case['schedule_identity']) and all(case['schedule_identity'].values())
    return passed


def native_prefixes(out,cases):
    checks={}
    for n,longer in zip(LENGTHS,LENGTHS[1:]):
        a,b=out/str(n)/'trace',out/str(longer)/'trace'
        h=cases[n]['hidden']
        for route in post.old.runner.ROUTES:
            for p in range(n):
                name=f'{route}_p{p}.bin'
                checks[f'{n}/{longer}/{name}']=digest(a/name)==digest(b/name)
            for layer in range(cases[n]['layers']):
                name=f'{route}_res{layer}.bin'
                x=base.read(a/name,(n+1)*h).astype('<f4')[:n*h]
                y=base.read(b/name,(longer+1)*h).astype('<f4')[:n*h]
                checks[f'{n}/{longer}/{name}']=x.tobytes()==y.tobytes()
    return checks


def measure(out,record,cases,offsets):
    measured={}
    for n,meta in cases.items():
        target=out/str(n)
        for kind in ('plain','trace'):
            if digest(target/kind/'ids.txt')!=meta['ids_sha256']:raise ValueError('prompt changed')
        plain,traced,schedule=post.measure(target,Path(record['references'])/str(n),meta,offsets,
                                          Path(record['plain_build']),Path(record['trace_build']))
        measured[str(n)]=dict(plain=plain,routes=traced,schedule_identity=schedule)
    return measured,native_prefixes(out,cases)


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    saved=dict(complete=False,passed=False,model_acceptance=False,sources=sources(),cases={},
               prepared=str(args.prepared.resolve()))
    try:
        saved['preparation_sha256']=digest(args.prepared.resolve()/'preparation.json')
        record,cases,offsets=verify_preparation(args.prepared.resolve())
        env=post.old.smoke.capture.environment()
        env.update(XILINX_XRT='/opt/xilinx/xrt',LD_LIBRARY_PATH='/opt/xilinx/xrt/lib',OPENBLAS_NUM_THREADS='1')
        with post.old.smoke.capture.device_lock():
            for n,meta in cases.items():
                for kind in ('plain','trace'):
                    target=out/str(n)/kind;target.mkdir(parents=True)
                    (target/'ids.txt').write_bytes((Path(record['references'])/str(n)/'ids.txt').read_bytes())
                    for route in post.old.runner.ROUTES:
                        base.memory_preflight(64);post.old.runner.device_available()
                        cmd=post.command(Path(record[kind+'_build']),target,meta,offsets,route)
                        print('running',n,kind,route,flush=True)
                        with (target/(route+'.log')).open('w') as log:
                            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
                plain,traced,schedule=post.measure(out/str(n),Path(record['references'])/str(n),meta,offsets,
                                                  Path(record['plain_build']),Path(record['trace_build']))
                saved['cases'][str(n)]=dict(plain=plain,routes=traced,schedule_identity=schedule)
                (out/'report.json').write_text(json.dumps(saved,indent=2,allow_nan=False)+'\n')
        saved['prefixes']=native_prefixes(out,cases)
        verify_preparation(Path(saved['prepared']))
        if digest(Path(saved['prepared'])/'preparation.json')!=saved['preparation_sha256']:
            raise ValueError('preparation changed during execution')
        saved.update(complete=True,passed=accept(saved['cases'],saved['prefixes']))
    except Exception as error:saved['error']=str(error);raise
    finally:(out/'report.json').write_text(json.dumps(saved,indent=2,allow_nan=False)+'\n')
    return 0 if saved['passed'] else 1


def audit(args):
    out=args.out.resolve();proof=dict(complete=False,passed=False,model_acceptance=False)
    try:
        saved=json.loads((out/'report.json').read_text())
        if not saved['complete'] or saved['model_acceptance'] is not False or saved['sources']!=sources():
            raise ValueError('incomplete run or changed validator')
        prepared=Path(saved['prepared'])
        if digest(prepared/'preparation.json')!=saved['preparation_sha256']:raise ValueError('preparation changed')
        record,cases,offsets=verify_preparation(prepared)
        measured,prefixes=measure(out,record,cases,offsets)
        if (measured,prefixes)!=(saved['cases'],saved['prefixes']):raise ValueError('measurements changed')
        passed=accept(measured,prefixes)
        if passed!=saved['passed']:raise ValueError('incorrect acceptance result')
        proof.update(complete=True,passed=passed,report_sha256=digest(out/'report.json'))
    except Exception as error:proof['error']=str(error);raise
    finally:(out/'audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    return 0 if proof['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for action,flags in [('prepare',('baseline','references','out')),('run',('prepared','out')),('audit',('out',))]:
        command=sub.add_parser(action)
        for flag in flags:command.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(dict(prepare=prepare,run=run,audit=audit)[args.action](args))
