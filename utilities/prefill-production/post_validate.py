"""Validate a new FP32 post-output export against fresh unpatched CLI controls.

Retained CPU references are immutable inputs, bound to their original audit.
Historical BF16 NPU hashes are not an oracle for the new arithmetic. Each trace
must instead reproduce a fresh unpatched run of the same current export.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

s=importlib.util.spec_from_file_location('_post_validation',Path(__file__).with_name('validate.py'))
old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
builder,base,digest=old.builder,old.base,old.digest


def inventory(out,route,tokens,layers,traced):
    expected=set(old.runner.capture_names(route,tokens,layers,traced))
    actual={p.name for pattern in (route+'_p*.bin',route+'_t*.bin',route+'_res*.bin',route+'-act.bin')
            for p in out.glob(pattern)}
    if actual!=expected:raise ValueError('incomplete or extra capture inventory: '+route)


def plain_identity(traced,plain,route,tokens,layers):
    if (set(traced)!=set(old.runner.capture_names(route,tokens,layers)) or
            set(plain)!=set(old.runner.capture_names(route,tokens,layers,False))):
        raise ValueError('incomplete or extra native identity inventory')
    result={n:traced[n]==h for n,h in plain.items()}
    if not all(result.values()):raise ValueError('fresh unpatched CLI identity failed')
    return result


def accept(plain,traced):
    if [r['route'] for r in plain]!=list(old.runner.ROUTES) or not all(r['passed'] for r in plain):
        raise ValueError('plain route coverage or numerical gate failed')
    return old.aggregate(traced,False)


def sources():
    return old.sources()|{str(Path(__file__).resolve()):digest(__file__)}


def fixture(baseline,kernels):
    report=baseline/'report.json';audit=baseline/'audit.json'
    saved=json.loads(report.read_text());proof=json.loads(audit.read_text())
    if not (saved['complete'] and saved['passed'] and proof['complete'] and proof['passed']
            and proof['report_sha256']==digest(report)):
        raise ValueError('retained reference lacks a bound passing audit')
    prepared=Path(saved['prepared'])
    meta=old.smoke.read_fixture(prepared,saved['preparation_sha256'])
    if (meta['layers']!=64 or meta['hidden']!=5120 or len(meta['ids'])!=20
            or meta['max_ctx']!=2048):raise ValueError('requires the full-depth short fixture')
    base.check_reference_coverage(meta);refs=prepared/'references'
    inputs={str(refs/n):h for n,h in meta['references'].items()}
    inputs[str(refs/'ids.txt')]=meta['ids_sha256']
    ids=[int(v) for v in (refs/'ids.txt').read_text().strip().split(',')]
    if ids!=meta['ids']:raise ValueError('reference prompt IDs differ')
    # Oracle arithmetic and model files must still match; old recipe/layout
    # source hashes stay historical because the post scratch ABI has changed.
    oracle=[base.ROOT/'utilities/prefill-slice/check.py',
            *(base.ROOT/'open_kernels/model'/n for n in
              ('replica_qwen35.py','replica_prompt.py','q4nx.py','compare_decode.py'))]
    model=Path(meta['model'])
    for path in [model/'model.q4nx',model/'config.json',model/'tokenizer.json',*oracle]:
        inputs[str(path)]=meta['sources'][str(path)]
    spec=base.spec_from_model_dir(model)
    current=json.loads((kernels/'manifest.json').read_text())
    selected=base.load_bundle(kernels,base.manifest(spec,max_ctx=meta['max_ctx']))
    post=selected['layer_types']['linear_attention'].get('post_output')
    if (selected['manifest_version']!=4 or selected['spec_hash']!=meta['spec_hash']
            or selected['layout']['hidden']!=meta['hidden'] or spec.vocab!=meta['vocab']
            or spec.real_vocab!=meta['real_vocab'] or len(selected['builds'])!=46
            or post!=dict(dtype='float32',offset=393216,bytes=24576)):
        raise ValueError('requires the current standard FP32 post-output export')
    # load_bundle validates semantics and binaries; current source identity is
    # checked separately, because historical exports are otherwise permitted.
    from recipes.cache import build_key
    if current['build_key']!=build_key(spec):raise ValueError('stale export source key')
    toolchain=json.loads((kernels/'toolchain.json').read_text())
    inputs.update({str(kernels/n):h for n,h in toolchain['sha256'].items()})
    for p in (report,audit,prepared/'preparation.json',refs/'reference.json',
              kernels/'manifest.json',kernels/'spec.json',kernels/'toolchain.json'):
        inputs[str(p)]=digest(p)
    base.verify(inputs)
    return meta|{'kernels':str(kernels)},refs,base.layout(spec,max_ctx=meta['max_ctx']),inputs


def plain_score(out,refs,meta,offsets,route):
    n=len(meta['ids']);inventory(out,route,n,meta['layers'],False)
    scores=[]
    for p in range(n+1):
        got=f'{route}_p{p}.bin' if p<n else route+'_t1.bin'
        ref=f'ref_p{p}.bin' if p<n else 'ref_cont_logits.bin'
        scores.append(dict(position=p,**base.logit_score(base.read(out/got,meta['vocab']),
                        base.read(refs/ref,meta['vocab']),meta['real_vocab'])))
    alias=digest(out/(route+'_t0.bin'))==digest(out/f'{route}_p{n-1}.bin')
    # Validate alias dimensions/finiteness even if a future hash provider changes.
    base.read(out/(route+'_t0.bin'),meta['vocab'])
    base.check_tokens((out/(route+'.log')).read_text(),meta['expected_tokens'])
    tail=base.final_residual((out/(route+'-act.bin')).read_bytes(),offsets,meta['hidden'])
    ref=base.read(refs/f'ref_cont_res{meta["layers"]-1}.bin',meta['hidden'])
    score=old.runner.trace.score_rows(tail[None,:],ref[None,:])[0]
    return dict(route=route,passed=alias and score['passed'] and all(r['passed'] for r in scores),
                logits=scores,last_prompt_alias=alias,continuation_residual=score,
                outputs={name:digest(out/name) for name in old.runner.capture_names(route,n,meta['layers'],False)},
                log_sha256=digest(out/(route+'.log')))


def command(build,out,meta,offsets,route):
    return base.native_command(build/'build/open_qwen36_cli',out,meta,offsets,route,old.runner.ROUTES[route])


def measure(out,refs,meta,offsets,plain_build,trace_build):
    plain=[];traced=[]
    for route in old.runner.ROUTES:
        row=plain_score(out/'plain',refs,meta,offsets,route)
        row['command']=command(plain_build,out/'plain',meta,offsets,route);plain.append(row)
        inventory(out/'trace',route,len(meta['ids']),meta['layers'],True)
        score=old.runner.score_route(out/'trace',refs,meta,offsets,route)
        traced.append(dict(route=route,first=score,
                      command=command(trace_build,out/'trace',meta,offsets,route),
                      identity=plain_identity(score['outputs'],row['outputs'],route,len(meta['ids']),meta['layers'])))
    return plain,traced,old.schedule_identity(traced)


def context(record):
    base.verify(record['sources'])
    for kind in ('plain','trace'):
        b=builder.verify(Path(record[kind+'_build']),False,plain=kind=='plain')
        if b['cli_sha256']!=record[kind+'_sha256']:raise ValueError('CLI changed')
    meta,refs,offsets,inputs=fixture(Path(record['baseline']),Path(record['kernels']))
    if inputs!=record['inputs']:raise ValueError('reference or export inputs changed')
    return meta,refs,offsets


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,passed=False,model_acceptance=False,sources=sources(),
           baseline=str(a.baseline.resolve()),kernels=str(a.kernels.resolve()),
           plain_build=str(a.plain_build.resolve()),trace_build=str(a.trace_build.resolve()),
           scope='64 layers, 20 prompt tokens plus continuation, three routes; no reset/long-prefix/performance acceptance')
    try:
        for kind in ('plain','trace'):
            r[kind+'_sha256']=builder.verify(Path(r[kind+'_build']),False,plain=kind=='plain')['cli_sha256']
        meta,refs,offsets,r['inputs']=fixture(Path(r['baseline']),Path(r['kernels']))
        env=old.smoke.capture.environment()
        env.update(XILINX_XRT='/opt/xilinx/xrt',LD_LIBRARY_PATH='/opt/xilinx/xrt/lib',OPENBLAS_NUM_THREADS='1')
        with old.smoke.capture.device_lock():
            for kind in ('plain','trace'):
                target=out/kind;target.mkdir();(target/'ids.txt').write_bytes((refs/'ids.txt').read_bytes())
                for route in old.runner.ROUTES:
                    base.memory_preflight(64);old.runner.device_available()
                    cmd=command(Path(r[kind+'_build']),target,meta,offsets,route)
                    print('running',kind,route,flush=True)
                    with (target/(route+'.log')).open('w') as log:
                        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
        r['plain'],r['routes'],r['schedule_identity']=measure(out,refs,meta,offsets,Path(r['plain_build']),Path(r['trace_build']))
        accept(r['plain'],r['routes']);context(r)
        r.update(complete=True,passed=True)
    except Exception as error:
        r['error']=str(error);raise
    finally:
        (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')


def audit(a):
    out=a.out.resolve();proof=dict(complete=False,passed=False,model_acceptance=False)
    try:
        r=json.loads((out/'report.json').read_text())
        if not r['complete'] or not r['passed'] or r['model_acceptance'] is not False or r['sources']!=sources():
            raise ValueError('incomplete result or changed validator')
        meta,refs,offsets=context(r)
        for kind in ('plain','trace'):
            if digest(out/kind/'ids.txt')!=meta['ids_sha256']:raise ValueError('prompt changed')
        plain,traced,schedule=measure(out,refs,meta,offsets,Path(r['plain_build']),Path(r['trace_build']))
        if (plain,traced,schedule)!=(r['plain'],r['routes'],r['schedule_identity']):
            raise ValueError('reported measurements differ')
        accept(plain,traced)
        proof.update(complete=True,passed=True,report_sha256=digest(out/'report.json'))
    except Exception as error:
        proof['error']=str(error);raise
    finally:
        (out/'audit.json').write_text(json.dumps(proof,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    q=sub.add_parser('run')
    for key in ('plain-build','trace-build','baseline','kernels','out'):q.add_argument('--'+key,type=Path,required=True)
    sub.add_parser('audit').add_argument('--out',type=Path,required=True)
    args=parser.parse_args();{'run':run,'audit':audit}[args.action](args)
