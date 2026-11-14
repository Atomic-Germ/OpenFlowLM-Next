"""Build a private current-arithmetic L3 projection control; pin compiler inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'open_kernels'))
from recipes.load import load_spec
from recipes.qwen35 import recipe


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


gen=load('_l3_projection_generator',ROOT/'open_kernels/designs/layer_x/gen_kernels.py')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def generated(spec):
    r=recipe(spec)
    if (spec.hidden!=5120 or spec.attn_q_width!=6144 or spec.attn_kv_width!=1024 or
            not r.attn.XN_F32 or not r.ffn.CONSISTENT_SUM or r.common.PER_CALL!=1):
        raise ValueError('requires current 27B FP32 consistent Q4 attention geometry')
    fs=gen.files(r)
    if 'gemv_q4_gy.cc' not in fs:raise ValueError('requires unmixed Q4 projection')
    out={n:fs[n] for n in ('dense_prep_f32.cc','gemv_q4_gy.cc')}
    out.update({n:(ROOT/'open_kernels/designs/gemv_q4'/n).read_text() for n in ('gemv_q4.h','gemv_tab.h')})
    out['design.py']='import sys\nsys.path.insert(0,'+repr(str(ROOT/'open_kernels'))+')\n'+(HERE/'design.py').read_text()
    return out


def dependencies(paths,directory):
    hashes={}
    for path in paths:
        tokens=shlex.split(path.read_text().replace('\\\n',' '))[1:]
        required={str(directory/n) for n in ('gemv_q4.h','gemv_tab.h')}
        if not required<=set(tokens):raise ValueError('compiler used wrong private header')
        for p in [path,*map(Path,tokens)]:hashes[str(p.resolve())]=digest(p)
    return hashes


def verify(build):
    r=json.loads((build/'build.json').read_text())
    if r.get('complete') is not True or r.get('model_acceptance') is not False:
        raise ValueError('incomplete projection build')
    for path,h in r['inputs'].items():
        if digest(path)!=h:raise ValueError('changed compiler input: '+path)
    expected=generated(load_spec(Path(r['spec'])))
    directory=build/'generated'
    if {p.name for p in directory.iterdir() if p.is_file()}!=set(expected):
        raise ValueError('changed generated inventory')
    for n,text in expected.items():
        if (directory/n).read_text()!=text:raise ValueError('changed generated source: '+n)
    for n,h in r['sha256'].items():
        if digest(build/n)!=h:raise ValueError('changed projection binary')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();spec=a.spec.resolve()
    if not out.is_relative_to(ROOT/'Models') or out==ROOT/'Models':raise ValueError('private output must be under Models')
    out.mkdir(parents=True,exist_ok=False);directory=out/'generated';directory.mkdir()
    fs=generated(load_spec(spec))
    for n,t in fs.items():(directory/n).write_text(t)
    inputs={str(p.resolve()):digest(p) for p in [Path(__file__),HERE/'design.py',Path(gen.__file__),spec,*directory.iterdir()]}
    subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(directory/'design.py'),str(out)],check=True)
    for path,h in inputs.items():
        if digest(path)!=h:raise ValueError('source changed during build')
    inputs.update(dependencies([out/'final.prj'/f'{n}.o.d' for n in ('dense_prep_f32','gemv_q4_gy')],directory))
    artifacts=[out/'final.xclbin',out/'insts.bin',out/'final.prj/input_with_addresses.mlir',
               *sorted((out/'final.prj').glob('elfs_main_core_*/*.elf'))]
    record=dict(complete=True,model_acceptance=False,spec=str(spec),inputs=inputs,
                sha256={str(p.relative_to(out)):digest(p) for p in artifacts})
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    verify(out)


if __name__=='__main__':main()
