"""Run private weighted FFN control/candidate serially from retained causal references."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
s=importlib.util.spec_from_file_location('_weighted_audit',Path(__file__).with_name('audit.py'))
audit=importlib.util.module_from_spec(s);s.loader.exec_module(audit)

def run(a):
    root=a.out.resolve();baseline=a.baseline.resolve()
    # Invalidate previous aggregate success even when a new preflight fails.
    result=dict(passed=False,capture_valid=False,model_acceptance=False)
    (root/'comparison.json').write_text(json.dumps(result)+'\n')
    meta,_=audit.trace.paired.validate_run(baseline)
    audit.base.memory_preflight(meta['layers'])
    for variant in ('control','candidate'):
        directory=root/variant
        audit.build(directory,variant)
        if (directory/'run').exists():raise FileExistsError(directory/'run')
    for variant in ('control','candidate'):
        directory=root/variant;out=directory/'run';out.mkdir()
        for name in ('reference.json','ids.txt',*meta['references']):shutil.copy2(baseline/name,out/name)
        audit.base.run(SimpleNamespace(out=out,cli=directory/'build/open_qwen36_cli'))
        # Numerical failure is preserved; identity must succeed before candidate execution.
        audit.trace.paired.validate_run(out)
        if variant=='control':audit.trace.require_identity(baseline,out,len(meta['ids']))
    status=audit.audit(a)
    p=root/'comparison.json';r=json.loads(p.read_text());r['evidence'][str(Path(__file__).resolve())]=audit.base.digest(Path(__file__))
    p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','baseline'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
