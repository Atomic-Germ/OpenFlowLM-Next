"""Audit private weighted FFN CLIs against exact controls and independent causal references."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('_weighted_trace_audit',ROOT/'utilities/prefill-residual-trace/audit.py')
trace=importlib.util.module_from_spec(s);s.loader.exec_module(trace)
base=trace.base

def inventory(directory,layers):
    expected={f'{r}_res{l}.bin' for r in trace.ROUTES for l in range(layers)}
    actual={p.name for p in directory.glob('*_res*.bin') if not p.name.startswith('ref_')}
    if actual!=expected:raise ValueError('incomplete or extra residual capture inventory')

def build(directory,variant):
    b=trace.verify_build(directory,variant)
    if not b.get('compiler_dependencies'):raise ValueError('missing compiler dependencies')
    base.verify(b['compiler_dependencies']);base.verify(b['kernel_inputs'])
    for path in b['kernel_inputs']:
        if Path(path).name == 'build.json':
            kernel=json.loads(Path(path).read_text())
            if not kernel['complete'] or kernel['mode']!='low':raise ValueError('invalid low kernel')
            for key in ('sources','generated','compiler_dependencies','binaries'):base.verify(kernel[key])
            base.verify(kernel['resources']['evidence'])
    return b

def audit(a):
    out=a.out.resolve();result=dict(passed=False,capture_valid=False,model_acceptance=False)
    try:
        baseline=a.baseline.resolve();bm,br=trace.paired.validate_run(baseline)
        n,layers,hidden=len(bm['ids']),bm['layers'],bm['hidden']
        reports={};measurements={};hashes={};evidence={}
        for variant in ('control','candidate'):
            directory=out/variant;run=directory/'run';b=build(directory,variant)
            meta,report=trace.paired.validate_run(run)
            if meta!=bm or report['cli_sha256']!=b['cli_sha256']:raise ValueError('reference/CLI mismatch')
            inventory(run,layers);rows=[];hh={}
            for layer in range(layers):
                ref=np.vstack((trace.read_trace(run/f'ref_res{layer}.bin',n,hidden),
                               trace.read_trace(run/f'ref_cont_res{layer}.bin',1,hidden)))
                for route in trace.ROUTES:
                    path=run/f'{route}_res{layer}.bin'
                    got=trace.read_trace(path,n+1,hidden)
                    rows.extend(dict(route=route,layer=layer,**r) for r in trace.score_rows(got,ref))
                    hh[path.name]=base.digest(path)
            pairs={str(l):hh[f'hybrid_res{l}.bin']==hh[f'block_major_res{l}.bin'] for l in range(layers)}
            if not all(pairs.values()):raise ValueError('hybrid residual schedule identity failed')
            measurements[variant]=dict(logits_passed=report['passed'],residuals_passed=all(r['passed'] for r in rows),
                hybrid_residual_identity=pairs,localization=trace.localize(rows,layers,n+1),
                routes=[dict(route=r['route'],passed=r['passed'],min_correlation=min(x['correlation'] for x in r['logits']),
                             continuation_residual_maxrel=r['continuation_last_residual_maxrel']) for r in report['routes']])
            hashes[variant]=hh;reports[variant]=report
            for p in (directory/'build-evidence.json',run/'report.json',run/'reference.json'):
                evidence[str(p)]=base.digest(p)
        control=out/'control/run';candidate=out/'candidate/run'
        control_identity=trace.require_identity(baseline,control,n)
        sequential=trace.paired.identity(control,candidate,n,('sequential',))
        sequential['act']=base.digest(control/'sequential-act.bin')==base.digest(candidate/'sequential-act.bin')
        sequential.update({f'res{l}':hashes['control'][f'sequential_res{l}.bin']==hashes['candidate'][f'sequential_res{l}.bin'] for l in range(layers)})
        if not all(sequential.values()):raise ValueError('sequential path changed')
        for p in (Path(__file__),Path(trace.__file__),Path(trace.paired.__file__),baseline/'report.json'):
            evidence[str(p.resolve())]=base.digest(p)
        result.update(capture_valid=True,passed=reports['candidate']['passed'] and measurements['candidate']['residuals_passed'],
            scope=f'{layers}-layer {n}-token causal prompt and one continuation, private paired-weight FFN',
            control_identity=control_identity,sequential_identity=sequential,measurements=measurements,
            trace_hashes=hashes,evidence=evidence)
    except Exception as e:
        result.update(passed=False,capture_valid=False,error=str(e));raise
    finally:(out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','baseline'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(audit(p.parse_args()))
