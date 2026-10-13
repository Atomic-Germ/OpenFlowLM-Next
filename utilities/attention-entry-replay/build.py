"""Pin a retained BF16 ax control or build a private FP32-entry ax experiment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from entry_common import ROOT,geometry,digest,source_hash,render_design
from recipes.load import load_spec
from recipes.cache import build_key
from recipes.manifest import manifest
from decode_bundle import load_bundle


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kernel-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['bf16','fp32'],required=True)
    a=ap.parse_args();spec=load_spec(a.kernel_dir/'spec.json')
    old=json.loads((a.kernel_dir/'manifest.json').read_text())
    m=load_bundle(a.kernel_dir,manifest(spec,max_ctx=old['max_ctx_default']))
    if m['build_key']!=build_key(spec):raise ValueError('baseline source identity changed')
    if a.out.resolve().is_relative_to(a.kernel_dir.resolve()):raise ValueError('keep experiment outside production export')
    a.out.mkdir(parents=True,exist_ok=True);record=a.out/'build.json';record.unlink(missing_ok=True)
    g=geometry(spec,a.mode)
    if a.mode=='bf16':
        for name in ('final.xclbin','insts.bin'):shutil.copyfile(a.kernel_dir/'ax'/name,a.out/name)
    else:
        design=a.out/'entry_ax.py';design.write_text(render_design(spec))
        env=dict(os.environ,OPEN_KERNELS_SPEC=str((a.kernel_dir/'spec.json').resolve()))
        subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(design),str(a.out)],env=env,check=True)
    meta=dict(mode=a.mode,geometry=g,production_build_key=m['build_key'],source_hash=source_hash(spec,a.mode),
              sha256={n:digest(a.out/n) for n in ('final.xclbin','insts.bin')})
    if a.mode=='fp32':meta['design_sha256']=digest(a.out/'entry_ax.py')
    record.write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
