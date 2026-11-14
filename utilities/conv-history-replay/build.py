"""Pin the FP32-entry control or build isolated FP32 conv history."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from history_common import ROOT,geometry,digest,source_hash,render_design,generated_sources,verify_generated
from history_common import entry
from recipes.load import load_spec
from recipes.cache import build_key
from recipes.manifest import manifest
from decode_bundle import load_bundle


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('kernel-dir','out','control-build'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--mode',choices=['bf16','fp32'],required=True);a=ap.parse_args()
    spec=load_spec(a.kernel_dir/'spec.json');old=json.loads((a.kernel_dir/'manifest.json').read_text())
    m=load_bundle(a.kernel_dir,manifest(spec,max_ctx=old['max_ctx_default']))
    if a.out.resolve()==a.kernel_dir.resolve() or a.out.resolve().is_relative_to(a.kernel_dir.resolve()):
        raise ValueError('experiment must be outside production export')
    a.out.mkdir(parents=True,exist_ok=True);record=a.out/'build.json';record.unlink(missing_ok=True)
    g=geometry(spec,a.mode)
    control=json.loads((a.control_build/'build.json').read_text())
    if (control['mode']!='fp32' or control['source_hash']!=entry.source_hash(spec,'fp32') or
            control['geometry']!=entry.geometry(spec,'fp32') or control['production_build_key']!=m['build_key']):
        raise ValueError('invalid FP32 entry control')
    if verify_generated(a.control_build/'generated',spec)!=control['generated_sha256']:
        raise ValueError('control generated sources mismatch')
    for name in ('final.xclbin','insts.bin'):
        if digest(a.control_build/name)!=control['sha256'][name]:raise ValueError('control binary mismatch')
    if a.mode=='bf16':
        for name in ('final.xclbin','insts.bin'):shutil.copyfile(a.control_build/name,a.out/name)
    else:
        generated=a.out/'generated';generated.mkdir(exist_ok=True)
        for name,source in generated_sources(spec).items():(generated/name).write_text(source)
        generated_hashes=verify_generated(generated,spec)
        design=a.out/'entry_lx.py';design.write_text(render_design(spec))
        env=dict(os.environ,OPEN_KERNELS_SPEC=str((a.kernel_dir/'spec.json').resolve()))
        subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(design),str(a.out)],env=env,check=True)
        if verify_generated(generated,spec)!=generated_hashes:raise ValueError('generated sources changed during build')
    meta=dict(mode=a.mode,geometry=g,production_build_key=m['build_key'],current_build_key=build_key(spec),
              source_hash=source_hash(spec,a.mode),control_sha256=control['sha256'],sha256={n:digest(a.out/n) for n in ('final.xclbin','insts.bin')})
    if a.mode=='fp32':
        meta['design_sha256']=digest(a.out/'entry_lx.py')
        meta['generated_sha256']=generated_hashes
    record.write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
