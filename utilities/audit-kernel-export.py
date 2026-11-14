"""Verify a complete export, its current source key and linked core program budgets."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'open_kernels'),str(ROOT/'open_kernels/model')]
from core_sizes import sections
from decode_bundle import load_bundle
from recipes.load import load_spec
from recipes.cache import build_key
from recipes.manifest import manifest
import hashlib


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit(export,build_root):
    spec=load_spec(export/'spec.json')
    recorded=json.loads((export/'manifest.json').read_text())
    expected=manifest(spec,max_ctx=recorded['max_ctx_default'])
    selected=load_bundle(export,expected)
    if selected['build_key']!=build_key(spec):raise ValueError('export does not match current sources')
    builds={}
    for name in selected['builds']:
        root=build_root/name
        for file in ('final.xclbin','insts.bin'):
            if digest(root/file)!=digest(export/name/file):raise ValueError(f'build/export mismatch: {name}/{file}')
        elfs=sorted((root/'final.prj').glob('elfs_main_core_*/*.elf'))
        if not elfs:raise ValueError(f'no retained core ELFs: {name}')
        sizes={p.name:sections(p.read_bytes()).get('.text',0) for p in elfs}
        if any(n<=0 or n>=16384 for n in sizes.values()):raise ValueError(f'program budget failure: {name}')
        builds[name]=dict(core_text=sizes,elf_sha256={str(p.resolve()):digest(p) for p in elfs})
    return dict(passed=True,build_key=selected['build_key'],spec_hash=selected['spec_hash'],builds=builds,
                binary_hashes=json.loads((export/'toolchain.json').read_text())['sha256'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('export','build-root','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();a.out.unlink(missing_ok=True)
    r=audit(a.export,a.build_root);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print('PASS',len(r['builds']),'sets,',len(r['binary_hashes']),'binary hashes, max core',
          max(n for b in r['builds'].values() for n in b['core_text'].values()),'bytes')


if __name__=='__main__':main()
