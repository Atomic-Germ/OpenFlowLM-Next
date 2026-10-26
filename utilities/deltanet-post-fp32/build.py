"""Private full-lx FP32 post-output experiment; production sources are untouched."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
LAYER=ROOT/'open_kernels/designs/layer_x'
POST=LAYER.parent/'dn_post'
for path in ('open_kernels','open_kernels/model'):
    sys.path.insert(0,str(ROOT/path))
from recipes.load import load_spec
from recipes.qwen35 import recipe
from recipes.cache import build_key
from recipes.manifest import manifest
from decode_bundle import load_bundle


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mode_check(mode):
    if mode not in ('control','fp32'):raise ValueError('unknown post experiment mode')


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('source anchor changed: '+old)
    return source.replace(old,new)


def geometry(spec,mode):
    mode_check(mode);r=recipe(spec);d=r.linear;l=r.layout
    if not (spec.hidden==5120 and spec.intermediate==17408 and d and d.VW==6144
            and d.FACTORED_READOUT and d.XN_F32 and d.CONV_F32 and r.ffn.CONSISTENT_SUM and not r.q8):
        raise ValueError('requires production wide dense consistent Q4 recipe')
    return dict(hidden=spec.hidden,legacy_act_bytes=l.A_BYTES,
                post_offset=l.A_BYTES if mode=='fp32' else l.A_OG,
                act_bytes=l.A_BYTES+(d.VW*4 if mode=='fp32' else 0),
                state_bytes=l.STATE_BYTES,post_elements=d.VW*(4 if mode=='fp32' else 2)//4096)


def post_source(mode):
    mode_check(mode)
    # Both modes compile the shared source. The same POST_F32 selection controls
    # its compiler flag, output argument, FIFO, DMA and projection preparation.
    return (POST/'post.cc').read_text()


def generated(spec,mode):
    geometry(spec,mode)
    loader=importlib.util.spec_from_file_location('_post_gen',LAYER/'gen_kernels.py')
    gen=importlib.util.module_from_spec(loader);loader.loader.exec_module(gen)
    return gen.files(recipe(spec)) | {'post.cc':post_source(mode),'post_copy.cc':(POST/'post_copy.cc').read_text()}


def helper(mode):
    mode_check(mode);s=(LAYER/'xlayer.py').read_text()
    s=replace_once(s,'HERE = Path(__file__).parent',f'HERE = Path({str(LAYER)!r})')
    s=replace_once(s,'POST = HERE.parent / "dn_post"','POST = Path(__file__).parent / "generated"')
    s=replace_once(s,'D = R.linear',
                   f'from dataclasses import replace\nD = replace(R.linear, POST_F32={mode == "fp32"})')
    return s


def source_hash(spec,mode):
    h=hashlib.sha256((build_key(spec)+mode).encode())
    for name,code in sorted(generated(spec,mode).items()):h.update(name.encode()+b'\0'+code.encode())
    for code in (helper(mode),(LAYER/'lx.py').read_text(),Path(__file__).read_text()):h.update(code.encode())
    return h.hexdigest()


def private_kernels(X,inc,t,directory):
    old=X.HERE
    try:
        X.HERE=directory
        return X.kernels([str(directory),*inc,str(old)],t)
    finally:X.HERE=old


def design(spec,mode):
    g=geometry(spec,mode);s=(LAYER/'lx.py').read_text()
    for old,new in [
        ('HERE = Path(__file__).parent',f'HERE = Path({str(LAYER)!r})'),
        ('import xlayer as XL', 'sys.path.insert(0, str(Path(__file__).parent))\nimport private_xlayer as XL'),
        ('import linear_fp32 as LF',f'import linear_fp32 as LF\nsys.path.insert(0, {str(HERE)!r})\nfrom build import private_kernels'),
        ('K = X.kernels(inc, t)','K = private_kernels(X, inc, t, Path(__file__).parent / "generated")')]:s=replace_once(s,old,new)
    s=replace_once(s,'D = R.linear',
                   f'from dataclasses import replace\nD = replace(R.linear, POST_F32={mode == "fp32"})')
    if mode=='fp32':
        s=replace_once(s,'from dataclasses import replace',
                       f'A_OG = {g["post_offset"]}\nA_BYTES = {g["act_bytes"]}\nfrom dataclasses import replace')
        s=replace_once(s,'OG_ELEMS = D.OG_ELEMS',f'OG_ELEMS = {g["post_elements"]}')
    return s+f'\nSPECIALIZE["srchash"] = {int(source_hash(spec,mode)[:8],16)}\n'


def dependencies(out):
    result={};parsed={}
    for p in sorted((out/'final.prj').glob('*.o.d')):
        paths=[Path(v).resolve() for v in shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1])]
        parsed[p.name]=paths
        for path in [p,*paths]:result[str(path)]=digest(path)
    for obj,source in (('post_fn','post'),('dense_prep_f32','dense_prep_f32'),('dense_prep','dense_prep'),('dnx_row','dnx_row'),('dnx_ofin','dnx_ofin')):
        paths=parsed.get(obj+'.o.d',[])
        if not any(p.name==obj+'.cc' for p in paths):raise ValueError('missing compiler source: '+obj)
        expected=out/'generated'/(source+'.cc')
        copied=[p for p in paths if p.name==obj+'.cc']
        if any(digest(p)!=digest(expected) for p in copied):raise ValueError('stale compiler TU: '+obj)
    for name in ('dnx_factored.h','dnx_precise.h'):
        if str((LAYER/name).resolve()) not in result:raise ValueError('missing production recurrence dependency')
    if str((LAYER.parent/'gemv_q4/gemv_tab.h').resolve()) not in result:raise ValueError('wrong Q4 header')
    return result


def verify(out):
    r=json.loads((out/'build.json').read_text())
    spec=load_spec(Path(r['kernels'])/'spec.json')
    if not r['complete'] or r['geometry']!=geometry(spec,r['mode']) or r['source_hash']!=source_hash(spec,r['mode']):
        raise ValueError('stale private build')
    for path,sha in r['inputs'].items():
        if digest(path)!=sha:raise ValueError('changed build input: '+path)
    for path,sha in r['artifacts'].items():
        if digest(out/path)!=sha:raise ValueError('changed build artifact: '+path)
    if dependencies(out)!=r['dependencies']:raise ValueError('changed compiler dependencies')
    return r


def main(a):
    out=a.out.resolve();kernels=a.kernels.resolve()
    if (not out.is_relative_to(ROOT/'Models') or out==ROOT/'Models' or
            out.is_relative_to(kernels) or kernels.is_relative_to(out)):
        raise ValueError('fresh separate private build under Models required')
    out.mkdir(parents=True,exist_ok=False);r=dict(complete=False,mode=a.mode,kernels=str(kernels))
    try:
        spec=load_spec(kernels/'spec.json');old=json.loads((kernels/'manifest.json').read_text())
        bundle=load_bundle(kernels,manifest(spec,max_ctx=old['max_ctx_default']))
        r.update(geometry=geometry(spec,a.mode),production_build_key=bundle['build_key'],source_hash=source_hash(spec,a.mode))
        gen=out/'generated';gen.mkdir()
        for n,code in generated(spec,a.mode).items():(gen/n).write_text(code)
        (out/'private_xlayer.py').write_text(helper(a.mode));(out/'full_lx.py').write_text(design(spec,a.mode))
        inputs=[Path(__file__),kernels/'spec.json',kernels/'manifest.json',kernels/'toolchain.json',out/'private_xlayer.py',out/'full_lx.py',*gen.iterdir()]
        r['inputs']={str(p):digest(p) for p in inputs}
        native_env=dict(os.environ,OPEN_KERNELS_SPEC=str(kernels/'spec.json'),LX_PART='0',LX_STOP='99')
        with (out/'build.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(out/'full_lx.py'),str(out)],
                           env=native_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if r['source_hash']!=source_hash(spec,a.mode) or any(digest(p)!=h for p,h in r['inputs'].items()):raise ValueError('sources changed during build')
        r.update(dependencies=dependencies(out),artifacts={n:digest(out/n) for n in ('final.xclbin','insts.bin')},complete=True)
    except Exception as error:r['error']=str(error);raise
    finally:(out/'build.json').write_text(json.dumps(r,indent=2)+'\n')
    verify(out)
    record=json.loads((out/"build.json").read_text());record["sha256"]=record["artifacts"]
    (out/"build.json").write_text(json.dumps(record,indent=2)+"\n")
    subprocess.run([sys.executable,str(ROOT/"utilities/attention-entry-replay/resources.py"),
                    "--build",str(out),"--out",str(out/"resources.json")],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernels',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['control','fp32'],required=True)
    main(p.parse_args())
