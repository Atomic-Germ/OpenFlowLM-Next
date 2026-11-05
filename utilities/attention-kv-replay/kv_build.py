"""Build an isolated attention mode and pin its sources, geometry and binaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from kv_common import ROOT,BASE,geometry
from replay_common import digest
from recipes.load import load_spec
from recipes.cache import build_key

HERE=Path(__file__).resolve().parent

def source_hash(spec,mode,tokens):
    h=hashlib.sha256(build_key(spec).encode())
    for p in sorted(HERE.glob('*.cc'))+sorted(HERE.glob('*.h'))+[HERE/n for n in ('kv_build.py','kv_probe.py','kv_common.py')]+[BASE/n for n in ('replay_common.py','probe_fin.cc','probe_debug.cc')]:
        h.update(p.name.encode()+b'\0'+p.read_bytes())
    h.update(f'{mode}:{tokens}'.encode())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['bf16','pair'],required=True);ap.add_argument('--tokens',type=int,default=3)
    a=ap.parse_args();spec=load_spec(a.spec);g=geometry(spec,a.tokens);mode=a.mode
    a.out.mkdir(parents=True,exist_ok=True);meta=a.out/'build.json';meta.unlink(missing_ok=True)
    env=dict(os.environ,OPEN_KERNELS_SPEC=str(a.spec.resolve()),KV_PAIR=str(int(mode=='pair')),REPLAY_TOKENS=str(a.tokens))
    subprocess.run([sys.executable,str(ROOT/'open_kernels/build_design.py'),str(HERE/'kv_probe.py'),str(a.out.resolve())],env=env,check=True)
    record=dict(mode=a.mode,tokens=a.tokens,geometry=g,source_hash=source_hash(spec,mode,a.tokens),
                production_build_key=build_key(spec),sha256={n:digest(a.out/n) for n in ('final.xclbin','insts.bin')})
    meta.write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
