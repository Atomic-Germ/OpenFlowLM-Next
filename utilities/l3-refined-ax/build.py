"""Private full-ax control and integrated NPU projection refinement; native six-buffer ABI."""
import argparse
import importlib.util
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
import hashlib


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def render(role, source=None):
    if role not in ('control','candidate'):raise ValueError('unknown build role')
    path=ROOT/'open_kernels/designs/layer_x/ax.py'
    s=path.read_text() if source is None else source
    pairs=[('HERE = Path(__file__).parent',f'HERE = Path({str(path.parent)!r})'),
           ('K = X.kernels(inc, t)','saved_here = X.HERE\n    X.HERE = Path(__file__).parent / "generated"\n    try:\n        K = X.kernels(inc + [str(saved_here)], t)\n    finally:\n        X.HERE = saved_here')]
    if role=='candidate':
        helper = (HERE/'worker.py.txt').read_text()
        pairs += [
            ('from aie.iron import CompileTime,', 'from aie.iron import Buffer, ExternalFunction, CompileTime,'),
            ('    t = X.types()', '    t = X.types()\n    t["tab"] = np.ndarray[(max(X.TAB_BYTES, 23040),), np.dtype[np.uint8]]'),
            ('    def main_body(win, xin, yout, *args):', helper+'\n    def main_body(win, xin, yout, low_y, prep_pair, gy_low, sum_pair, *args):'),
            ('X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS, 2 * Q_PC + 2 * KV_PC, "attn", fp32=D.XN_F32)',
             'refined_bands(win, xin, yout, B, K, low_y, prep_pair, gy_low, sum_pair)'),
            ('XL.main_workers(main_body, of_w, of_x, of_y, t, K)',
             'XL.main_workers(main_body, of_w, of_x, of_y, t, K, extra=lambda c: [Buffer(t["y"], name=f"low_y{c}"), prep_pair, gy_low, sum_pair])')]
    for old,new in pairs:
        if s.count(old)!=1:raise ValueError('changed full-ax anchor: '+old)
        s=s.replace(old,new)
    return s+'\nSPECIALIZE["srchash"] = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8], 16)\n'


def generated(spec, role):
    if role not in ('control','candidate'):raise ValueError('unknown build role')
    r=recipe(spec)
    if (spec.hidden!=5120 or spec.num_layers!=64 or r.q8 or not r.attn.KV_PAIR or not r.attn.XN_F32 or
        not r.ffn.XM_F32 or not r.ffn.CONSISTENT_SUM):raise ValueError('requires current 27B Q4 precision geometry')
    p=ROOT/'open_kernels/designs/layer_x/gen_kernels.py'
    s=importlib.util.spec_from_file_location('_refined_ax_gen',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    fs=m.files(r)
    if role=='candidate':fs.update({p.name:p.read_text() for p in [*HERE.glob('*.cc'),*HERE.glob('*.h')]})
    return fs


def sources(spec):
    return {str(p.resolve()):digest(p) for p in source_files(spec)+[Path(__file__),HERE/'worker.py.txt',*HERE.glob('*.cc'),*HERE.glob('*.h'),ROOT/'open_kernels/build_design.py',ROOT/'open_kernels/ironutil.py']}


def resources(out):
    prj=out/'final.prj';files=sorted(prj.glob('elfs_main_core_*/*.elf'))
    sizes={p.name:sections(p.read_bytes())['.text'] for p in files};l1={}
    mlir=(prj/'input_with_addresses.mlir').read_text()
    for line in mlir.splitlines():
        m=re.search(r'aie.buffer\(%tile_(\d+_[23])\).*address = (\d+).*memref<(\d+)x(bf16|f32|i32|ui8)>',line)
        if m:
            tile,addr,n,ty=m.groups();end=int(addr)+int(n)*{'bf16':2,'f32':4,'i32':4,'ui8':1}[ty]
            l1[tile]=max(l1.get(tile,0),end)
    if len(sizes)!=15 or len(l1)!=15 or max(sizes.values())>=16384 or max(l1.values())>65536:
        raise ValueError('full-ax resource gate failed')
    if mlir.count('stack_size = 6144 : i32')!=15:raise ValueError('changed core stacks')
    return dict(text=sizes,l1_end=l1,stack=6144)


def verify(out):
    r=json.loads((out/'build.json').read_text());spec=load_spec(Path(r['spec']))
    if r.get('complete') is not True or r.get('model_acceptance') is not False or r['sources']!=sources(spec):
        raise ValueError('incomplete or changed full-ax build')
    if (out/'refined_ax.py').read_text()!=render(r['role']):raise ValueError('changed refined-ax design')
    fs=generated(spec,r['role'])
    if {p.name for p in (out/'generated').iterdir()}!=set(fs):raise ValueError('changed generated inventory')
    for n,s in fs.items():
        if (out/'generated'/n).read_text()!=s:raise ValueError('changed generated translation unit')
    for p,h in r['inputs'].items():
        if digest(p)!=h:raise ValueError('changed compiler input: '+p)
    for n,h in r['sha256'].items():
        if digest(out/n)!=h:raise ValueError('changed full-ax artifact')
    if r['resources']!=resources(out) or r['build_key']!=build_key(spec):raise ValueError('changed build resources/key')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--role',choices=['control','candidate'],required=True)
    a=p.parse_args();out=a.out.resolve();specpath=a.spec.resolve();spec=load_spec(specpath);fs=generated(spec,a.role)
    if not out.is_relative_to(ROOT/'Models') or out==ROOT/'Models':raise ValueError('private output must be under Models')
    out.mkdir(parents=True,exist_ok=False);(out/'generated').mkdir()
    for n,s in fs.items():(out/'generated'/n).write_text(s)
    design=out/'refined_ax.py';design.write_text(render(a.role));src=sources(spec)
    inputs={str(p):digest(p) for p in [specpath,design,*list((out/'generated').iterdir())]}
    subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(design),str(out)],
                   env=dict(os.environ,OPEN_KERNELS_SPEC=str(specpath)),check=True)
    if sources(spec)!=src:raise ValueError('sources changed during build')
    deps=list((out/'final.prj').glob('*.o.d'))
    if not deps:raise ValueError('missing compiler dependencies')
    compiled=set()
    for d in deps:
        for f in [d,*map(Path,shlex.split(d.read_text().replace('\\\n',' '))[1:])]:inputs[str(f.resolve())]=digest(f)
        n=d.name.removesuffix('.o.d')+'.cc'
        if n in fs:
            if (out/'final.prj'/n).read_text()!=fs[n]:raise ValueError('stale compiled generated TU')
            compiled.add(n)
    if a.role=='candidate' and not {'refine_prep.cc','refine_low.cc','refine_sum.cc'}<=compiled:raise ValueError('missing refinement TUs')
    if not {'dense_prep_f32.cc','gemv_q4_gy.cc','gemv_q4_gms.cc'}<=compiled:raise ValueError('missing main-core production TUs')
    artifacts=[out/'final.xclbin',out/'insts.bin',out/'final.prj/input_with_addresses.mlir',*sorted((out/'final.prj').glob('elfs_main_core_*/*.elf'))]
    r=dict(complete=True,model_acceptance=False,role=a.role,spec=str(specpath),sources=src,inputs=inputs,build_key=build_key(spec),
           resources=resources(out),sha256={str(p.relative_to(out)):digest(p) for p in artifacts})
    (out/'build.json').write_text(json.dumps(r,indent=2)+'\n');verify(out)


if __name__=='__main__':main()
