"""Run selected sublayer captures with independent references in a new directory."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
s=importlib.util.spec_from_file_location('_sublayer_check',Path(__file__).with_name('check.py'))
check=importlib.util.module_from_spec(s);s.loader.exec_module(check)

def run(a):
 meta=json.loads((a.baseline/'reference.json').read_text());check.base.memory_preflight(meta['layers'])
 build=check.trace.verify_build(a.build.resolve(),'paired-sublayer-trace');check.base.verify(build['compiler_dependencies'])
 prior=json.loads((a.baseline/'trace-audit.json').read_text())
 if not prior['capture_valid']:raise ValueError('invalid baseline')
 check.base.verify(prior['evidence'])
 a.out.mkdir(parents=True,exist_ok=False)
 for name in ['reference.json','ids.txt',*meta['references']]:shutil.copy2(a.baseline/name,a.out/name)
 check.base.run(SimpleNamespace(out=a.out,cli=a.build/'build/open_qwen36_cli'))
 return check.check(SimpleNamespace(build=a.build,baseline=a.baseline,run=a.out,out=a.out/'sublayer-audit.json'))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('build','baseline','out'):p.add_argument('--'+name,type=Path,required=True)
 raise SystemExit(run(p.parse_args()))
