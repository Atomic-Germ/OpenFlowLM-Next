"""Build a private full-lx candidate or pin an unchanged production control."""
import argparse,json,os,shutil,subprocess,sys,shlex
from pathlib import Path
from full_layer_common import ROOT,geometry,source_hash,generated_sources,render_design,verify_generated,digest
from recipes.load import load_spec
from recipes.manifest import manifest
from decode_bundle import load_bundle


def verify_dependencies(build):
    result={}
    for name in ('vcopy','pass1','delta','row','ofin'):
        p=build/'final.prj'/f'dnx_{name}.o.d'
        deps=shlex.split(p.read_text().replace('\\\n',''))
        if not all(str(build/'generated'/n) in deps for n in ('dnx.h','precise.h')):
            raise ValueError('compiled object did not use private headers: '+name)
        result[p.name]=digest(p)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('kernel-dir','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=['control','factored'],required=True);p.add_argument('--shared',action='store_true');a=p.parse_args()
    if a.out.resolve()==a.kernel_dir.resolve() or a.out.resolve().is_relative_to(a.kernel_dir.resolve()):
        raise ValueError('private build must be outside production export')
    if a.mode=='control' and a.shared:raise ValueError('control cannot change helper linkage')
    a.out=a.out.resolve()
    spec=load_spec(a.kernel_dir/'spec.json');old=json.loads((a.kernel_dir/'manifest.json').read_text())
    m=load_bundle(a.kernel_dir,manifest(spec,max_ctx=old['max_ctx_default']))
    a.out.mkdir(parents=True,exist_ok=True);record=a.out/'build.json';record.unlink(missing_ok=True)
    meta=dict(mode=a.mode,shared=a.shared,geometry=geometry(spec),production_build_key=m['build_key'],source_hash=source_hash(spec,a.shared))
    if a.mode=='control':
        for n in ('final.xclbin','insts.bin'):shutil.copyfile(a.kernel_dir/'lx'/n,a.out/n)
    else:
        directory=a.out/'generated';directory.mkdir(exist_ok=True)
        for n,s in generated_sources(spec,a.shared).items():(directory/n).write_text(s)
        hashes=verify_generated(directory,spec,a.shared)
        design=a.out/'full_lx.py';design.write_text(render_design(spec,a.shared))
        meta.update(generated_sha256=hashes,design_sha256=digest(design))
        (a.out/'attempt.json').write_text(json.dumps(meta,indent=2)+'\n')
        env=dict(os.environ,OPEN_KERNELS_SPEC=str((a.kernel_dir/'spec.json').resolve()))
        subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(design),str(a.out)],env=env,check=True)
        if verify_generated(directory,spec,a.shared)!=hashes:raise ValueError('sources changed during build')
        if digest(design)!=meta['design_sha256'] or source_hash(spec,a.shared)!=meta['source_hash']:raise ValueError('design changed during build')
        meta['dependency_sha256']=verify_dependencies(a.out)
        sys.path.insert(0,str(ROOT/'utilities'))
        from core_sizes import sections
        sizes={p.stem:sections(p.read_bytes())['.text'] for p in (a.out/'final.prj').glob('elfs_main_core_*/*.elf')}
        resource=dict(passed=bool(sizes) and max(sizes.values())<16384,core_text=sizes)
        (a.out/'resource-attempt.json').write_text(json.dumps(resource,indent=2)+'\n')
        if not resource['passed']:raise ValueError('full-layer program memory exceeded')
    meta['sha256']={n:digest(a.out/n) for n in ('final.xclbin','insts.bin')}
    record.write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
