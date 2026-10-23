"""Capture weighted-candidate sublayers and audit them against retained native outputs."""
import argparse
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace
s=importlib.util.spec_from_file_location('_weighted_capture_check',Path(__file__).with_name('check.py'))
check=importlib.util.module_from_spec(s);s.loader.exec_module(check)

def run(a):
 out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 check.build(a.build.resolve(),'weighted-sublayer')
 check.build(a.baseline.parent.resolve(),'candidate')
 meta,_=check.trace.paired.validate_run(a.baseline.resolve())
 check.base.memory_preflight(meta['layers'])
 out.mkdir(parents=True)
 for name in ('reference.json','ids.txt',*meta['references']):shutil.copy2(a.baseline/name,out/name)
 check.base.run(SimpleNamespace(out=out,cli=a.build/'build/open_qwen36_cli'))
 return check.check(SimpleNamespace(build=a.build,baseline=a.baseline,run=out,out=out/'sublayer-audit.json'))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('build','baseline','out'):p.add_argument('--'+name,type=Path,required=True)
 raise SystemExit(run(p.parse_args()))
