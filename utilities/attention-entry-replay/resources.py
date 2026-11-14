"""Audit the experimental ax's linked programs and compiler-assigned L1 buffers."""
import argparse
import json
import re
import math
import sys
from pathlib import Path
from entry_common import ROOT,digest
sys.path.insert(0,str(ROOT/'utilities'))
from core_sizes import sections


def audit(build):
    record=json.loads((build/'build.json').read_text())
    for n,h in record['sha256'].items():
        if digest(build/n)!=h:raise ValueError('binary hash mismatch')
    prj=build/'final.prj';elfs=sorted(prj.glob('elfs_main_core_*/*.elf'))
    sizes={re.search(r'(\d+_\d+)$',p.stem)[1]:sections(p.read_bytes())['.text'] for p in elfs}
    l1={};source=prj/'input_with_addresses.mlir'
    for line in source.read_text().splitlines():
        if not re.search(r'aie.buffer\(%tile_\d+_[2-5]\)',line):continue
        m=re.search(r'aie.buffer\(%tile_(\d+_[2-5])\).*address = (\d+).*memref<([\dx]+)x(bf16|f32|i32|ui8)>',line)
        if not m:raise ValueError('unsupported compute-buffer declaration')
        tile,addr,dims,ty=m.groups();n=math.prod(int(v) for v in dims.split('x'))
        end=int(addr)+n*{'bf16':2,'f32':4,'i32':4,'ui8':1}[ty]
        l1[tile]=max(l1.get(tile,0),end)
    if not sizes or set(sizes)!=set(l1):raise ValueError('incomplete compute-core coverage')
    if max(sizes.values())>=16384 or max(l1.values())>65536:raise ValueError('memory budget exceeded')
    return dict(passed=True,build=record,core_text=sizes,l1_buffer_end=l1,
                evidence_sha256={str(p.resolve()):digest(p) for p in [build/'build.json',source,*elfs]})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--build',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.unlink(missing_ok=True);r=audit(a.build);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print('PASS',len(r['core_text']),'cores, max text',max(r['core_text'].values()),'L1 end',max(r['l1_buffer_end'].values()))
