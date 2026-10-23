"""Audit production lx/ax consistent Q4 sum selection and unchanged factored recurrence."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
LAYER=ROOT/'open_kernels/designs/layer_x'
GEMV=ROOT/'open_kernels/designs/gemv_q4'
sys.path[:0]=[str(ROOT/'open_kernels'),str(ROOT/'open_kernels/model')]
from decode_bundle import load_bundle
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes.cache import build_key


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dependencies(name):
    result={obj:[GEMV/'gemv_tab.h',GEMV/'gemv_q4.h']
            for obj in ('dense_prep_f32','dense_prep','gemv_q4_gy','gemv_q4_gms')}
    if name=='lx':
        result.update({'dnx_'+n:[LAYER/'dnx_factored.h',LAYER/'dnx_precise.h']
                       for n in ('vcopy','pass1','delta','row','ofin')})
    elif name!='ax':raise ValueError('expected lx or ax')
    return result


def audit_compilation(build_root):
    hashes={}
    for name in ('lx','ax'):
        prj=build_root/name/'final.prj';source=prj/'dense_prep_f32.cc'
        text=source.read_text()
        if (text.count('#define GEMV_Q4_CONSISTENT_SUM 1')!=1
                or text.count('gemv_q4_prep_f32_direct_blocks(')!=1):
            raise ValueError('compiled FP32 prep selection is missing: '+name)
        hashes[str(source.resolve())]=digest(source)
        for obj,headers in dependencies(name).items():
            dep=prj/(obj+'.o.d')
            tokens=shlex.split(dep.read_text().replace('\\\n',''))
            for path in headers:
                if str(path) not in tokens:raise ValueError('compiler used wrong header: '+str(dep))
                hashes[str(path)]=digest(path)
            hashes[str(dep.resolve())]=digest(dep)
    return hashes


def audit(export,build_root):
    spec=load_spec(export/'spec.json');old=json.loads((export/'manifest.json').read_text())
    selected=load_bundle(export,manifest(spec,max_ctx=old['max_ctx_default']))
    if selected['build_key']!=build_key(spec):raise ValueError('export source identity mismatch')
    if selected['layout'].get('q4_activation_sum')!='quantized_fp32':raise ValueError('missing arithmetic identity')
    hashes=audit_compilation(build_root)
    for name in ('lx','ax'):
        for filename in ('final.xclbin','insts.bin'):
            source=build_root/name/filename;dest=export/name/filename
            if digest(source)!=digest(dest):raise ValueError('build/export binary mismatch')
            hashes[str(source.resolve())]=digest(source)
    return dict(passed=True,build_key=selected['build_key'],sha256=hashes)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('export','build-root','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.out.unlink(missing_ok=True)
    result=audit(a.export.resolve(),a.build_root.resolve())
    a.out.write_text(json.dumps(result,indent=2)+'\n');print('PASS compiled lx/ax consistent sum and factored recurrence')
