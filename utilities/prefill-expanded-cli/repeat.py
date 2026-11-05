"""Validate two separately captured requests on one resident expanded engine."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
s=importlib.util.spec_from_file_location('_expanded_resident',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(s);s.loader.exec_module(runner)
prep,base,digest=runner.prep,runner.base,runner.digest
sections=prep.builder.repeat.sections


def aggregate(rows):
    if [r['route'] for r in rows]!=list(runner.ROUTES) or any(not r['repeat_identity'] or not all(r['repeat_identity'].values()) for r in rows):
        raise ValueError('incomplete routes or changed repeated captures')
    return dict(passed=True,numerical_passed=all(r[key]['passed'] for r in rows for key in ('first','second')))


def context(baseline):
    saved,record,meta,offsets=runner.check_run(baseline)
    audited=json.loads((baseline/'audit.json').read_text())
    if not audited['complete'] or audited['report_sha256']!=digest(baseline/'report.json') or audited['passed']!=saved['passed']:
        raise ValueError('single-request baseline has no complete independent audit')
    return saved,record,meta,offsets


def measure_route(out,baseline,prepared,meta,offsets,route):
    raw=out/(route+'-requests.log');parts=sections(raw.read_text())
    scores=[]
    for prefix,text in zip((route,route+'_repeat'),parts):
        if (out/(prefix+'.log')).read_text()!=text:raise ValueError('request log extraction changed')
        scores.append(runner.original.score_route(out,prepared/'references',meta,offsets,prefix))
    identity=runner.require_identity(baseline/'expanded',out,len(meta['ids']),meta['layers'],(route,))
    repeated={name:digest(out/name)==digest(out/name.replace(route,route+'_repeat',1))
              for name in runner.original.capture_names(route,len(meta['ids']),meta['layers'])}
    if not all(repeated.values()):raise ValueError('repeated captures differ')
    return dict(route=route,first=scores[0],second=scores[1],baseline_identity=identity,
                repeat_identity=repeated,raw_log_sha256=digest(raw))


def sources():
    return runner.evidence()|{str(Path(__file__).resolve()):digest(__file__),str(Path(prep.builder.repeat.__file__).resolve()):digest(prep.builder.repeat.__file__)}


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,passed=False,numerical_passed=False,model_acceptance=False,routes=[])
    try:
        baseline=a.baseline.resolve();saved,record,meta,offsets=context(baseline)
        prepared=Path(saved['prepared']);build=record['builds']['expanded-repeat'];bp=Path(build['directory'])
        r.update(baseline=str(baseline),baseline_report_sha256=digest(baseline/'report.json'),
                 preparation_sha256=digest(prepared/'preparation.json'),cli_sha256=build['cli_sha256'],sources=sources())
        (out/'ids.txt').write_bytes((prepared/'references/ids.txt').read_bytes())
        with runner.device_lock():
            for route,extra in runner.ROUTES.items():
                base.memory_preflight(meta['layers']);runner.original.device_available()
                cmd=base.native_command(bp/'build/open_qwen36_cli',out,meta,offsets,route,extra+['--twice'])
                print('repeated resident requests:',route,flush=True)
                with (out/(route+'-requests.log')).open('w') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=runner.environment(),check=True,timeout=1800)
                for prefix,text in zip((route,route+'_repeat'),sections((out/(route+'-requests.log')).read_text())):
                    (out/(prefix+'.log')).write_text(text)
                row=measure_route(out,baseline,prepared,meta,offsets,route);row['command']=cmd;r['routes'].append(row)
                print(route,len(row['repeat_identity']),'captures exact',flush=True)
                (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        base.verify(r['sources']);r.update(complete=True,**aggregate(r['routes']),scope='normal Core::reset between identical resident requests; not checkpoint or full-buffer clearing acceptance')
    except Exception as error:r.update(complete=False,passed=False,error=str(error));raise
    finally:(out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    return 0


def audit(a):
    out=a.out.resolve();r=dict(complete=False,passed=False,model_acceptance=False)
    try:
        saved=json.loads((out/'report.json').read_text())
        if not saved['complete'] or saved['sources']!=sources():raise ValueError('incomplete repeat run or changed validator')
        base.verify(saved['sources']);baseline=Path(saved['baseline']);original,record,meta,offsets=context(baseline)
        prepared=Path(original['prepared']);bp=Path(record['builds']['expanded-repeat']['directory'])
        if (saved['baseline_report_sha256']!=digest(baseline/'report.json') or
            saved['preparation_sha256']!=digest(prepared/'preparation.json') or
            saved['cli_sha256']!=record['builds']['expanded-repeat']['cli_sha256']):raise ValueError('repeat provenance changed')
        if [row['route'] for row in saved['routes']]!=list(runner.ROUTES):raise ValueError('incomplete repeated routes')
        measured=[]
        for row in saved['routes']:
            route=row['route'];got=measure_route(out,baseline,prepared,meta,offsets,route)
            got['command']=base.native_command(bp/'build/open_qwen36_cli',out,meta,offsets,route,runner.ROUTES[route]+['--twice'])
            if got!=row:raise ValueError('repeat measurements or command changed')
            measured.append(got)
        values=aggregate(measured)
        if any(saved[k]!=v for k,v in values.items()) or saved['model_acceptance'] is not False:
            raise ValueError('incorrect repeat aggregate')
        r.update(complete=True,**values,report_sha256=digest(out/'report.json'))
    except Exception as error:r['error']=str(error);raise
    finally:(out/'audit.json').write_text(json.dumps(r,indent=2)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','audit'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--baseline',type=Path)
    a=p.parse_args()
    if a.command=='run' and a.baseline is None:p.error('--baseline required')
    raise SystemExit(run(a) if a.command=='run' else audit(a))
