"""Capture three combined routes serially, requiring exact native identity."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
s=importlib.util.spec_from_file_location('_combined_sublayer_check',Path(__file__).with_name('check.py'))
check=importlib.util.module_from_spec(s);s.loader.exec_module(check)
runner=check.runner;digest=check.digest


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(complete=False,passed=False,model_acceptance=False,commands=[])
    try:
        baseline=a.baseline.resolve();build=a.build.resolve()
        original,b,meta,offsets=check.context(baseline,build)
        refs=Path(original['prepared'])/'references'
        (out/'ids.txt').write_bytes((refs/'ids.txt').read_bytes())
        env={k:v for k,v in os.environ.items() if not k.startswith('OFLM_OPEN_')}
        env.update(OMP_NUM_THREADS='2',OMP_WAIT_POLICY='PASSIVE',OFLM_OPEN_ATTN_BLOCK='0',
                   OFLM_OPEN_LAYER_MAJOR='1',OFLM_OPEN_STAGE_MAJOR='0')
        with (runner.prep.ROOT/'Models/qwen38-27b/.resident-validation.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for route,extra in runner.ROUTES.items():
                runner.base.memory_preflight(meta['layers']);runner.device_available()
                command=runner.base.native_command(build/'build/open_qwen36_cli',out,meta,offsets,route,extra)
                result['commands'].append(command);print('capturing',route,flush=True)
                with (out/(route+'.log')).open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
                identity=runner.require_identity(baseline/'combined',out,len(meta['ids']),meta['layers'],(route,))
                print(route,len(identity),'baseline captures exact',flush=True)
        check.inventory(out)
        measured=runner.audit_variant(out,refs,meta,offsets)
        result.update(complete=True,passed=True,numerical_passed=measured['passed'],measurements=measured,
                      baseline=str(baseline),build=str(build),cli_sha256=b['cli_sha256'],
                      baseline_report_sha256=digest(baseline/'report.json'),runner_sha256=digest(Path(__file__)),
                      captures={name:digest(out/name) for name in check.expected_captures()})
    except Exception as error:
        result.update(complete=False,passed=False,error=str(error));raise
    finally:(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','build','baseline'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
