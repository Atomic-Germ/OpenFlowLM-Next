"""Bind fresh expanded builds to exact retained controls, replay and independent references."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


builder=load('_expanded_build',Path(__file__).with_name('build.py'))
runner=builder.repeat.runner
base,digest=runner.base,runner.base.digest


def check_replay(report,audit):
    gates=('complete','control_valid','slot_identity','projection_passed','host_passed','conditioned_passed','residual_passed')
    if report!=audit or report.get('dispatches')!=64 or any(report.get(k) is not True for k in gates):
        raise ValueError('attention replay has no complete accepted audit')


def verify_build(directory,variant):
    b=runner.trace.verify_build(directory,variant)
    if not b.get('compiler_dependencies'):raise ValueError('missing compiler dependencies')
    base.verify(b['compiler_dependencies']);base.verify(b['kernel_inputs'])
    if set(b['low_kernels'])!=set(builder.roles_for(variant)):raise ValueError('wrong low kernel inventory')
    for role,name in b['low_kernels'].items():
        path=Path(name)
        if b['kernel_inputs'].get(str(path/'build.json'))!=digest(path/'build.json'):
            raise ValueError('unbound low-weight build')
        builder.verify_kernel(path,role)
    return b


def require_repeat_sources(one,twice):
    def inventory(directory):
        return {str(p.relative_to(directory)):digest(p) for p in directory.rglob('*')
                if p.is_file() and p.relative_to(directory)!=Path('cli.cpp')}
    if inventory(one)!=inventory(twice):raise ValueError('repeat build changed engine source inventory/arithmetic')


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,native_validated=False,model_acceptance=False,builds={},inputs={},artifacts={})
    def pin(path):r['inputs'][str(path.resolve())]=digest(path)
    try:
        for variant in builder.VARIANTS:
            directory=a.builds.resolve()/variant;b=verify_build(directory,variant)
            r['builds'][variant]=dict(directory=str(directory),cli_sha256=b['cli_sha256'])
            pin(directory/'build-evidence.json')
        # Only CLI request-capture plumbing may differ in the repeat executable.
        one=a.builds.resolve()/'expanded/source/src/open_qwen36'
        twice=a.builds.resolve()/'expanded-repeat/source/src/open_qwen36'
        require_repeat_sources(one,twice)
        replay=a.replay.resolve();rep=json.loads((replay/'report.json').read_text());aud=json.loads((replay/'audit.json').read_text())
        check_replay(rep,aud);base.verify(rep['evidence']);base.verify(rep['artifacts'])
        r['inputs'].update(rep['evidence']);r['inputs'].update(rep['artifacts'])
        prep_paths=[Path(p) for p in rep['evidence'] if Path(p).name=='preparation.json']
        if len(prep_paths)!=1:raise ValueError('ambiguous replay preparation')
        replay_preparation=json.loads(prep_paths[0].read_text())
        if not replay_preparation['complete']:raise ValueError('incomplete replay preparation')
        base.verify(replay_preparation['inputs']);base.verify(replay_preparation['artifacts'])
        r['inputs'].update(replay_preparation['inputs']);r['inputs'].update(replay_preparation['artifacts'])
        pin(replay/'report.json');pin(replay/'audit.json')
        baseline=a.baseline.resolve();old=json.loads((baseline/'report.json').read_text());audit=json.loads((baseline/'audit.json').read_text())
        if not old['complete'] or not audit['complete'] or audit['report_sha256']!=digest(baseline/'report.json'):
            raise ValueError('baseline lacks complete audit')
        prepared=Path(old['prepared']);record,meta,offsets=runner.preparation(prepared)
        if old['preparation_sha256']!=digest(prepared/'preparation.json') or old['runner_sha256']!=digest(runner.__file__):
            raise ValueError('baseline provenance changed')
        if meta['layers']!=64 or meta['hidden']!=5120 or len(meta['ids'])!=20:raise ValueError('wrong baseline extent')
        for variant in ('paired','weighted','combined'):
            measured=runner.audit_variant(baseline/variant,prepared/'references',meta,offsets)
            saved=old['variants'][variant]
            if any(saved.get(k)!=v for k,v in measured.items()) or saved['cli_sha256']!=record['builds'][variant]['cli_sha256']:
                raise ValueError('baseline measurements/build changed')
            for route in measured['routes']:
                r['inputs'].update({str(baseline/variant/name):h for name,h in route['outputs'].items()})
                pin(baseline/variant/(route['route']+'.log'))
        if old['passed']!=old['variants']['combined']['passed'] or audit['passed']!=old['passed']:
            raise ValueError('baseline aggregate changed')
        r['inputs'].update(record['inputs']);r['inputs'].update(record['artifacts'])
        pin(prepared/'preparation.json');pin(baseline/'report.json');pin(baseline/'audit.json')
        refs=out/'references';refs.mkdir()
        for name in ('reference.json','ids.txt',*meta['references']):
            shutil.copy2(prepared/'references'/name,refs/name);r['artifacts'][str(refs/name)]=digest(refs/name)
        for p in (Path(__file__),Path(builder.__file__),Path(runner.__file__),Path(runner.prep.__file__)):pin(p)
        # The only independent reference remains unchanged; acceptance is deferred.
        base.verify(r['inputs']);base.verify(r['artifacts'])
        r.update(complete=True,baseline=str(baseline),baseline_prepared=str(prepared),
                 baseline_passed=old['passed'],replay=str(replay),reference_files=len(meta['references']),
                 manifest_matches_current=True,repeat_engine_exact=True,
                 pending=['fresh combined control exact capture identity','expanded full-depth independent residual/logit gates',
                          'expanded repeated resident/reset captures','long-prefix and production acceptance'])
    except Exception as error:r['error']=str(error);raise
    finally:(out/'preparation.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('out','builds','baseline','replay'):p.add_argument('--'+k,type=Path,required=True)
    raise SystemExit(prepare(p.parse_args()))
