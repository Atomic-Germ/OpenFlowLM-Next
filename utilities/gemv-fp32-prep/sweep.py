"""Run and score all isolated Qwen3.5 layers serially with experimental lx/ax.

Each process holds one layer only. This is conditioned replay, not full-model
propagation: inputs at each position are the retained previous-layer outputs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__)
for name in ('model-dir','kernel-dir','fixture','pools','candidate-root','out'):
    ap.add_argument('--'+name,type=Path,required=True)
a=ap.parse_args()
if a.out.resolve() in [p.resolve() for p in (a.model_dir, a.kernel_dir, a.fixture, a.pools, a.candidate_root)]:
    raise ValueError('sweep output must be separate from its inputs')
a.out.mkdir(parents=True,exist_ok=True)
# Never leave a stale successful summary if the new sweep is interrupted.
summary=a.out/'results.json';summary.unlink(missing_ok=True)
m=json.loads((a.kernel_dir/'manifest.json').read_text())
sys.path.insert(0, str(ROOT/'open_kernels/model'))
from decode_reference import load_reference
meta=load_reference(a.fixture)
if not 0 < meta['layers'] <= len(m['layers']):
    raise ValueError('fixture layer coverage exceeds the manifest')
results=[]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',HARNESS_TIMEOUT_MS='30000')
for layer,kind in enumerate(m['layers'][:meta['layers']]):
    out=a.out/f'layer{layer}';out.mkdir(exist_ok=True)
    candidate=a.candidate_root/('ax-direct' if kind=='full_attention' else 'lx-direct')
    command=[sys.executable,str(ROOT/'utilities/qwen35_layer_replay.py'),
             '--model-dir',str(a.model_dir),'--kernel-dir',str(a.kernel_dir),
             '--fixture',str(a.fixture),'--pools',str(a.pools),'--out',str(out),
             '--candidate',str(candidate),'--layer',str(layer)]
    with (out/'prepare.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'hardware.log').open('w') as log:
        subprocess.run([str(ROOT/'open_kernels/harness/out/run_kernel'),str(out/'run.cfg')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'score.log').open('w') as log:
        status=subprocess.run(command+['--score'],env=env,stdout=log,stderr=subprocess.STDOUT)
    # A malformed/missing score is an error, not an omitted layer.
    result=json.loads((out/'score.json').read_text())
    if status.returncode not in (0,1) or (status.returncode==0) != result['passed']:
        raise RuntimeError(f'inconsistent score exit status at layer {layer}')
    results.append(result)
    print(f'layer {layer}: {"PASS" if result["passed"] else "FAIL"}',flush=True)
r=dict(passed=all(r['passed'] for r in results),layers=len(results),tokens=meta['tokens'],
       scope='Isolated layer replay conditioned on retained inputs; not full-model propagation',results=results)
summary.write_text(json.dumps(r,indent=2)+'\n')
sys.exit(0 if r['passed'] else 1)
