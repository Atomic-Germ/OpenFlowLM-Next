"""Build a private arithmetic probe and record complete compiler/binary provenance."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_refined_probe_build',HERE/'build.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if not out.is_relative_to(b.ROOT/'Models'):raise ValueError('private output must be under Models')
    out.mkdir(parents=True,exist_ok=False)
    src=[Path(__file__),HERE/'build.py',HERE/'refine_prep.cc',HERE/'refine_sum.cc',*(HERE/'probe').iterdir(),*HERE.glob('*.h')]
    inputs={str(p.resolve()):b.digest(p) for p in src}
    for p in src:
        if p.name not in ('probe.py','build.py'):(out/p.name).write_bytes(p.read_bytes())
    inputs.update({str(p):b.digest(p) for p in out.iterdir()})
    subprocess.run([sys.executable,str(b.ROOT/'open_kernels/build_design.py'),str(out/'design.py'),str(out)],check=True)
    deps=list((out/'final.prj').glob('*.o.d'))
    if not deps:raise ValueError('no compiler provenance')
    for d in deps:
        for p in [d,*map(Path,shlex.split(d.read_text().replace('\\\n',' '))[1:])]:inputs[str(p.resolve())]=b.digest(p)
    elfs=list((out/'final.prj').glob('elfs_main_core_*/*.elf'))
    if len(elfs)!=1:raise ValueError('wrong core count')
    size=b.sections(elfs[0].read_bytes())['.text']
    if size>=16384:raise ValueError('program overflow')
    for p,h in inputs.items():
        if b.digest(p)!=h:raise ValueError('changed compiler input')
    artifacts=[out/'final.xclbin',out/'insts.bin',out/'final.prj/input_with_addresses.mlir',*elfs]
    (out/'build.json').write_text(json.dumps(dict(complete=True,model_acceptance=False,inputs=inputs,text=size,
        sha256={str(p):b.digest(p) for p in artifacts}),indent=2)+'\n')

if __name__=='__main__':main()
