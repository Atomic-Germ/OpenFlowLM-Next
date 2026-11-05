"""Fresh production attention and extracted FFN host bridges with full dependencies."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('_attention_ffn_bridge',ROOT/'utilities/prefill-ffn-replay/bridge.py')
ffn=importlib.util.module_from_spec(s);s.loader.exec_module(ffn)
digest=ffn.digest


def build(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,commands=[])
    try:
        ffn.build(out/'ffn')
        sources=[ROOT/'utilities/prefill-boundaries/bridge.cpp',ROOT/'src/open_qwen36/block_host.cpp',Path(__file__)]
        r['sources']={str(p):digest(p) for p in sources}
        deps=set();objects=[]
        for src in sources[:2]:
            obj=out/(src.stem+'.o');dep=obj.with_suffix('.d');objects.append(str(obj))
            cmd=['g++','-std=c++17','-O2','-fPIC','-mavx2','-mfma','-ffp-contract=off','-fopenmp',
                 '-I'+str(ROOT/'src'),'-I'+str(ROOT/'src/include'),'-MD','-MF',str(dep),'-c',str(src),'-o',str(obj)]
            r['commands'].append(cmd);subprocess.run(cmd,check=True)
            deps.update(Path(n).resolve() for n in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]))
        cmd=['g++','-shared','-fopenmp',*objects,'-o',str(out/'attention.so')]
        r['commands'].append(cmd);subprocess.run(cmd,check=True)
        r.update(complete=True,dependencies={str(p):digest(p) for p in sorted(deps)},
                 binary_sha256=digest(out/'attention.so'),compiler=subprocess.check_output(['g++','--version'],text=True))
    finally:(out/'build.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
