"""Run a private trace in a fresh directory using retained independent references."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

s = importlib.util.spec_from_file_location('_residual_trace_audit', Path(__file__).with_name('audit.py'))
audit = importlib.util.module_from_spec(s)
s.loader.exec_module(audit)


def run(a):
    # Check residency before copying or launching anything. Never reuse captures.
    meta = json.loads((a.baseline/'reference.json').read_text())
    audit.base.memory_preflight(meta['layers'])
    audit.verify_build(a.build.resolve(), 'paired-residual-trace')
    control = audit.verify_build(a.baseline_build.resolve(), 'paired')
    meta, report = audit.paired.validate_run(a.baseline.resolve())
    if report['cli_sha256'] != control['cli_sha256']:
        raise ValueError('baseline CLI differs from paired build')
    a.out.mkdir(parents=True, exist_ok=False)
    for name in ['reference.json', 'ids.txt', *meta['references']]:
        shutil.copy2(a.baseline/name, a.out/name)
    # Numeric failure is expected for the retained long-prefix fixtures. A failed
    # process or preflight raises; only completed measurements reach the audit.
    audit.base.run(SimpleNamespace(out=a.out, cli=a.build/'build/open_qwen36_cli'))
    return audit.audit(SimpleNamespace(build=a.build, baseline_build=a.baseline_build,
                                      baseline=a.baseline, run=a.out, out=a.out/'trace-audit.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build','baseline-build','baseline','out'):
        p.add_argument('--'+name, type=Path, required=True)
    raise SystemExit(run(p.parse_args()))
