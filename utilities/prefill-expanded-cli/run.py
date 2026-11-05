"""Validate fresh expanded prefill propagation against exact combined controls."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess

s=importlib.util.spec_from_file_location('_expanded_preparation',Path(__file__).with_name('prepare.py'))
prep=importlib.util.module_from_spec(s);s.loader.exec_module(prep)
original=prep.runner
base,digest=prep.base,prep.digest
ROUTES=original.ROUTES
VARIANTS=('combined','expanded')
require_identity=original.require_identity


def check_preparation(record):
    if (not record.get('complete') or not record.get('manifest_matches_current') or
        set(record.get('builds',{}))!=set(prep.builder.VARIANTS)):
        raise ValueError('incomplete expanded preparation/build inventory')


def preparation(directory):
    record=json.loads((directory/'preparation.json').read_text());check_preparation(record)
    base.verify(record['inputs']);base.verify(record['artifacts'])
    for variant,build in record['builds'].items():
        b=prep.verify_build(Path(build['directory']),variant)
        if b['cli_sha256']!=build['cli_sha256']:raise ValueError('prepared CLI changed')
    one=Path(record['builds']['expanded']['directory'])/'source/src/open_qwen36'
    twice=Path(record['builds']['expanded-repeat']['directory'])/'source/src/open_qwen36'
    prep.require_repeat_sources(one,twice)
    old,meta,offsets=original.preparation(Path(record['baseline_prepared']))
    refs=directory/'references';expected=Path(record['baseline_prepared'])/'references'
    for name in ('reference.json','ids.txt',*meta['references']):
        if digest(refs/name)!=digest(expected/name):raise ValueError('independent reference changed: '+name)
    if meta['layers']!=64 or len(meta['ids'])!=20 or meta['hidden']!=5120:
        raise ValueError('requires full-depth 20-token fixture')
    baseline=Path(record['baseline']);saved=json.loads((baseline/'report.json').read_text())
    audit=json.loads((baseline/'audit.json').read_text())
    if (not saved['complete'] or not audit['complete'] or audit['report_sha256']!=digest(baseline/'report.json')
        or Path(saved['prepared'])!=Path(record['baseline_prepared'])
        or saved['variants']['combined']['cli_sha256']!=old['builds']['combined']['cli_sha256']):
        raise ValueError('unbound combined baseline')
    return record,meta,offsets


def environment():
    env={k:v for k,v in os.environ.items() if not k.startswith('OFLM_OPEN_')}
    env.update(OMP_NUM_THREADS='2',OMP_WAIT_POLICY='PASSIVE',OFLM_OPEN_ATTN_BLOCK='0',
               OFLM_OPEN_LAYER_MAJOR='1',OFLM_OPEN_STAGE_MAJOR='0')
    return env


@contextmanager
def device_lock():
    with (prep.ROOT/'Models/qwen38-27b/.resident-validation.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original.device_available()
        yield


def evidence():
    paths=(Path(__file__),Path(prep.__file__),Path(original.__file__),Path(base.__file__),Path(original.trace.__file__))
    return {str(p.resolve()):digest(p) for p in paths}


def capture_variant(variant,out,prepared,record,meta,offsets):
    directory=out/variant;directory.mkdir()
    (directory/'ids.txt').write_bytes((prepared/'references/ids.txt').read_bytes())
    build=record['builds'][variant];bp=Path(build['directory'])
    prep.verify_build(bp,variant);commands=[]
    for route,extra in ROUTES.items():
        base.memory_preflight(meta['layers']);original.device_available()
        cmd=base.native_command(bp/'build/open_qwen36_cli',directory,meta,offsets,route,extra)
        commands.append(cmd);print('running',variant,route,flush=True)
        with (directory/(route+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment(),check=True,timeout=1800)
        score=original.score_route(directory,prepared/'references',meta,offsets,route)
        print(variant,route,'logits',score['logits_passed'],'residuals',sum(x['passed'] for x in score['residuals']), '/',len(score['residuals']),flush=True)
    report=original.audit_variant(directory,prepared/'references',meta,offsets)
    report.update(cli_sha256=build['cli_sha256'],commands=commands)
    return report


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,passed=False,model_acceptance=False,variants={},controls={})
    try:
        prepared=a.prepared.resolve();record,meta,offsets=preparation(prepared)
        r.update(prepared=str(prepared),preparation_sha256=digest(prepared/'preparation.json'),sources=evidence())
        with device_lock():
            for variant in VARIANTS:
                base.memory_preflight(meta['layers'])
                r['variants'][variant]=capture_variant(variant,out,prepared,record,meta,offsets)
                if variant=='combined':
                    r['controls']['combined']=require_identity(Path(record['baseline'])/'combined',out/variant,len(meta['ids']),meta['layers'])
                else:
                    r['controls']['sequential']=require_identity(out/'combined',out/variant,len(meta['ids']),meta['layers'],('sequential',))
                (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        base.verify(r['sources'])
        r.update(complete=True,passed=r['variants']['expanded']['passed'],
                 scope='64-layer 20-token prompt plus continuation; repeat/reset is a separate check')
    except Exception as error:r.update(complete=False,passed=False,error=str(error));raise
    finally:(out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    return 0 if r['passed'] else 1


def check_run(out):
    saved=json.loads((out/'report.json').read_text())
    if not saved['complete'] or set(saved['variants'])!=set(VARIANTS) or saved['sources']!=evidence():
        raise ValueError('incomplete run or changed validator')
    base.verify(saved['sources']);prepared=Path(saved['prepared'])
    if digest(prepared/'preparation.json')!=saved['preparation_sha256']:raise ValueError('preparation changed')
    record,meta,offsets=preparation(prepared)
    for variant in VARIANTS:
        measured=original.audit_variant(out/variant,prepared/'references',meta,offsets);stored=saved['variants'][variant]
        if any(stored.get(k)!=v for k,v in measured.items()):raise ValueError('reported measurements differ')
        bp=Path(record['builds'][variant]['directory'])
        commands=[base.native_command(bp/'build/open_qwen36_cli',out/variant,meta,offsets,route,extra) for route,extra in ROUTES.items()]
        if stored['commands']!=commands or stored['cli_sha256']!=record['builds'][variant]['cli_sha256']:
            raise ValueError('run commands/build changed')
    controls=dict(combined=require_identity(Path(record['baseline'])/'combined',out/'combined',len(meta['ids']),meta['layers']),
                  sequential=require_identity(out/'combined',out/'expanded',len(meta['ids']),meta['layers'],('sequential',)))
    if controls!=saved['controls'] or saved['passed']!=saved['variants']['expanded']['passed'] or saved['model_acceptance'] is not False:
        raise ValueError('incorrect aggregate/control result')
    return saved,record,meta,offsets


def audit(a):
    out=a.out.resolve();r=dict(complete=False,passed=False,model_acceptance=False)
    try:
        saved,*_=check_run(out)
        r.update(complete=True,passed=saved['passed'],report_sha256=digest(out/'report.json'))
    except Exception as error:r['error']=str(error);raise
    finally:(out/'audit.json').write_text(json.dumps(r,indent=2)+'\n')
    return 0 if r['passed'] else 1


def preflight(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,native_validated=False,model_acceptance=False)
    try:
        prepared=a.prepared.resolve();record,meta,_=preparation(prepared)
        available=None;memory_ready=False
        try:available=base.memory_preflight(meta['layers']);memory_ready=True
        except RuntimeError as error:r['memory_error']=str(error)
        r.update(complete=True,memory_ready=memory_ready,available_bytes=available,prepared=str(prepared),
                 preparation_sha256=digest(prepared/'preparation.json'),sources=evidence(),builds=record['builds'])
    except Exception as error:r['error']=str(error);raise
    finally:(out/'preflight.json').write_text(json.dumps(r,indent=2)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','run','audit'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--prepared',type=Path)
    a=p.parse_args()
    if a.command!='audit' and a.prepared is None:p.error('--prepared required')
    raise SystemExit(dict(preflight=preflight,run=run,audit=audit)[a.command](a))
