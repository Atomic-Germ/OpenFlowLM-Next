"""Record linked program sizes and the compiler's sequential L1 allocation."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core_sizes import sections
import kv_common  # establish shared replay imports
from replay_common import digest


def inspect(build):
    meta=json.loads((build/'build.json').read_text());g=meta['geometry']
    prj=build/'final.prj';files=sorted(prj.glob('elfs_main_core_*/*.elf'))
    sizes={p.name:sections(p.read_bytes())['.text'] for p in files}
    source=prj/'input_with_addresses.mlir';l1={}
    for line in source.read_text().splitlines():
        m=re.search(r'aie.buffer\(%tile_(\d+_3)\).*address = (\d+).*memref<(\d+)x(bf16|f32|i32|ui8)>',line)
        if m:
            tile,addr,n,ty=m.groups();end=int(addr)+int(n)*{'bf16':2,'f32':4,'i32':4,'ui8':1}[ty]
            l1[tile]=max(l1.get(tile,0),end)
    if len(sizes)!=g['cores'] or len(l1)!=g['cores']:raise ValueError('incomplete core resource coverage')
    if max(sizes.values())>=16384 or max(l1.values())>65536:raise ValueError('core memory budget exceeded')
    data_bytes=g['tokens']*(g['input_row']+g['kv_row_bytes']+g['qw']*4+g['cores']*g['debug_core'])
    return dict(build=meta,core_text=sizes,l1_address_end=l1,data_bo_bytes=data_bytes,
                evidence_sha256={str(p.resolve()):digest(p) for p in [build/'build.json',source,*files]})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    out=a.root/'build-audit.json';out.unlink(missing_ok=True)
    r={m:inspect(a.root/(m+'-build')) for m in ('bf16','pair')}
    out.write_text(json.dumps(r,indent=2)+'\n')
    for m,v in r.items():print(m,'text',max(v['core_text'].values()),'L1',max(v['l1_address_end'].values()),'BO bytes',v['data_bo_bytes'])


if __name__=='__main__':main()
