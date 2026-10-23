"""Build private recurrence binaries and pin sources; never rewrite production."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from recurrence_common import ROOT,HERE,digest,geometry,sources
from recipes.load import load_spec
from recipes.manifest import manifest
from decode_bundle import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['control','triple','factored'],required=True);a=p.parse_args()
    if a.out.resolve()==a.kernel_dir.resolve() or a.out.resolve().is_relative_to(a.kernel_dir.resolve()):
        raise ValueError('private build must be outside production export')
    spec=load_spec(a.kernel_dir/'spec.json');old=json.loads((a.kernel_dir/'manifest.json').read_text())
    m=load_bundle(a.kernel_dir,manifest(spec,max_ctx=old['max_ctx_default']));g=geometry(spec)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'build.json').unlink(missing_ok=True)
    directory=a.out/'source';directory.mkdir(exist_ok=True)
    for name,code in sources(a.mode).items():(directory/name).write_text(code)
    hashes={n:digest(directory/n) for n in sources(a.mode)}
    subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(directory/'design.py'),str(a.out)],check=True)
    if hashes!={n:digest(directory/n) for n in sources(a.mode)}:raise ValueError('sources changed during build')
    (a.out/'build.json').write_text(json.dumps(dict(mode=a.mode,geometry=g,production_build_key=m['build_key'],
        source_sha256=hashes,sha256={n:digest(a.out/n) for n in ('final.xclbin','insts.bin')}),indent=2)+'\n')


if __name__=='__main__':main()
