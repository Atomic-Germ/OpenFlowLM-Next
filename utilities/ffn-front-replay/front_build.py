"""Build and audit the isolated FFN probe with production Q4 preparation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
GEMV=ROOT/'open_kernels/designs/gemv_q4'
LN=ROOT/'open_kernels/designs/ln'
sys.path.insert(0,str(ROOT/'open_kernels'))
from recipes.load import load_spec
from recipes.cache import source_files,build_key
from recipes.qwen36moe import consistent_q4_sum


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prep_flags(mode):
    if mode not in (0,1,2):raise ValueError('FRONT_FP32 must be 0, 1 or 2')
    return ['-DFRONT_FP32='+str(int(mode!=0))]+(['-DGEMV_Q4_CONSISTENT_SUM=1'] if mode==2 else [])


def validate_spec(spec,mode):
    prep_flags(mode)
    if spec.family!='qwen35' or 'ffn' in spec.q8_roles or spec.hidden%1024:
        raise ValueError('requires qwen35 Q4 FFN, hidden divisible by 1024')
    if mode==2 and not consistent_q4_sum(spec):raise ValueError('unvalidated production geometry')


def sources(spec):
    return sorted(set(source_files(spec)+list(HERE.glob('*.cc'))+[HERE/'probe.py',HERE/'front_build.py']))


def source_hashes(spec):return {str(p.resolve()):digest(p) for p in sources(spec)}


def audit_compilation(build):
    prj=build/'final.prj';hashes={}
    for name in ('front_prep','front_gms','front_emit','ln_nr32'):
        expected=(LN if name=='ln_nr32' else HERE)/(name+'.cc');actual=prj/(name+'.cc')
        if actual.read_bytes()!=expected.read_bytes():raise ValueError('compiled source drift: '+name)
        dep=prj/(name+'.o.d');tokens=shlex.split(dep.read_text().replace('\\\n',''))
        headers=[LN/'ln.h'] if name=='ln_nr32' else [GEMV/'gemv_q4.h',GEMV/'gemv_tab.h']
        if any(str(h) not in tokens for h in headers):raise ValueError('compiler used wrong header: '+name)
        for p in [actual,dep,*headers]:hashes[str(p.resolve())]=digest(p)
    return hashes


def verify_build(build,spec):
    r=json.loads((build/'build.json').read_text())
    validate_spec(spec,r['mode'])
    if r['mode']!=2 or r['build_key']!=build_key(spec) or r['prep_flags']!=prep_flags(2):
        raise ValueError('probe does not match production arithmetic')
    if r['sources']!=source_hashes(spec) or r['compiled']!=audit_compilation(build):
        raise ValueError('probe source identity mismatch')
    if r['sha256']!={n:digest(build/n) for n in ('final.xclbin','insts.bin')}:
        raise ValueError('probe binary hash mismatch')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out=a.out.resolve();spec=load_spec(a.spec);validate_spec(spec,2)
    if not a.out.is_relative_to(ROOT/'Models'):raise ValueError('build must be under Models')
    a.out.mkdir(parents=True,exist_ok=True);record=a.out/'build.json';record.unlink(missing_ok=True)
    before=source_hashes(spec)
    env=dict(os.environ,OPEN_KERNELS_SPEC=str(a.spec.resolve()),FRONT_FP32='2')
    subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(HERE/'probe.py'),str(a.out)],env=env,check=True)
    if before!=source_hashes(spec):raise ValueError('source changed during compilation')
    r=dict(mode=2,build_key=build_key(spec),prep_flags=prep_flags(2),sources=before,
           compiled=audit_compilation(a.out),sha256={n:digest(a.out/n) for n in ('final.xclbin','insts.bin')})
    record.write_text(json.dumps(r,indent=2)+'\n');print('PASS compiled production FFN probe')


if __name__=='__main__':main()
