"""Build isolated current paired attention using the production worker verbatim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'open_kernels'),str(ROOT/'utilities')]
from recipes.load import load_spec
from recipes.qwen35 import recipe
from recipes.cache import source_files,build_key
from core_sizes import sections


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sources(spec):
    paths=source_files(spec)+[HERE/'build.py',HERE/'design.py',ROOT/'open_kernels/ironutil.py',ROOT/'open_kernels/build_design.py']
    return {str(p.resolve()):digest(p) for p in paths}


def geometry(spec):
    a=recipe(spec).attn
    if (spec.hidden!=5120 or spec.num_layers!=64 or
        (a.NH,a.KVH,a.HD,a.NHL,a.ACORES,a.RB,a.E_A)!=(24,4,256,4,6,1,2048) or
        not a.KV_PAIR or not a.VEXP):raise ValueError('requires current 27B paired attention')
    return dict(tokens=240,input_row=61440,kv_row=8192,output_row=24576,cores=6,stack=6144)


def resources(out):
    prj=out/'final.prj';files=sorted(prj.glob('elfs_main_core_*/*.elf'))
    sizes={p.name:sections(p.read_bytes())['.text'] for p in files};l1={}
    for line in (prj/'input_with_addresses.mlir').read_text().splitlines():
        m=re.search(r'aie.buffer\(%tile_(\d+_3)\).*address = (\d+).*memref<(\d+)x(bf16|f32|i32|ui8)>',line)
        if m:
            tile,addr,n,ty=m.groups();end=int(addr)+int(n)*{'bf16':2,'f32':4,'i32':4,'ui8':1}[ty]
            l1[tile]=max(l1.get(tile,0),end)
    if len(sizes)!=6 or len(l1)!=6 or max(sizes.values())>=16384 or max(l1.values())>65536:
        raise ValueError('incomplete or overflowing attention core resources')
    return dict(text=sizes,l1_end=l1,stack=6144,data_bo_bytes=240*(61440+8192+24576))


def verify(out):
    r=json.loads((out/'build.json').read_text());spec=load_spec(Path(r['spec']))
    if (r.get('complete') is not True or r.get('model_acceptance') is not False or
        r['sources']!=sources(spec) or r['geometry']!=geometry(spec) or r['build_key']!=build_key(spec)):
        raise ValueError('incomplete or changed attention build')
    for k in ('inputs','sources'):
        for p,h in r[k].items():
            if digest(p)!=h:raise ValueError('changed compiler input: '+p)
    for n,h in r['sha256'].items():
        if digest(out/n)!=h:raise ValueError('changed attention binary: '+n)
    if r['resources']!=resources(out):raise ValueError('changed attention resources')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();specpath=a.spec.resolve();spec=load_spec(specpath)
    g=geometry(spec);src=sources(spec)
    if not out.is_relative_to(ROOT/'Models') or out==ROOT/'Models':raise ValueError('private build must be under Models')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OPEN_KERNELS_SPEC=str(specpath))
    subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(HERE/'design.py'),str(out)],env=env,check=True)
    if sources(spec)!=src:raise ValueError('sources changed during build')
    deps=list((out/'final.prj').glob('*.o.d'))
    required={'attn_meta','attn_q','attn_pair_k','attn_pair_v','attn_init','attn_pair_step','attn_pair_step_new','attn_pair_fin'}
    if not required<={p.name.removesuffix('.o.d') for p in deps}:raise ValueError('missing compiler dependencies')
    inputs={str(specpath):digest(specpath)}
    for d in deps:
        for f in [d,*map(Path,shlex.split(d.read_text().replace('\\\n',' '))[1:])]:inputs[str(f.resolve())]=digest(f)
    # IRON copies each translation unit into final.prj before compiling it.
    # Bind that copy and require byte identity with the production source.
    for n in required:
        copied=out/'final.prj'/f'{n}.cc'
        original=ROOT/'open_kernels/designs/attn'/f'{n}.cc'
        if str(copied) not in inputs or digest(copied)!=digest(original):
            raise ValueError('compiler did not use production attention source: '+n)
    expected={str(ROOT/'open_kernels/designs/attn'/n) for n in ('attn.h','attn_pair.h')}
    if not expected<=set(inputs):raise ValueError('compiler used different attention headers')
    artifacts=[out/'final.xclbin',out/'insts.bin',out/'final.prj/input_with_addresses.mlir',
               *sorted((out/'final.prj').glob('elfs_main_core_*/*.elf'))]
    r=dict(complete=True,model_acceptance=False,spec=str(specpath),sources=src,inputs=inputs,
           geometry=g,build_key=build_key(spec),resources=resources(out),
           sha256={str(f.relative_to(out)):digest(f) for f in artifacts})
    (out/'build.json').write_text(json.dumps(r,indent=2)+'\n');verify(out)


if __name__=='__main__':main()
