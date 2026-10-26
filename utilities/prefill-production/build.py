"""Build capture-only shared-runtime CLIs; never patch projection arithmetic."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


trace = load('_production_trace', ROOT/'utilities/prefill-residual-trace/build.py')
repeat_tool = load('_production_repeat', ROOT/'utilities/prefill-combined-cli/repeat.py')
base, digest = repeat_tool.base, repeat_tool.digest


def instrument(cpp, cli, repeat):
    cpp, cli = trace.patch(cpp, cli)
    return cpp, repeat_tool.patch(cli) if repeat else cli


def build(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    r = dict(complete=False, repeat=a.repeat, native_validated=False, model_acceptance=False, commands=[])
    try:
        original = ROOT/'src/open_qwen36'
        dst = out/'source/src/open_qwen36'
        shutil.copytree(original, dst, ignore=shutil.ignore_patterns('build', '__pycache__'))
        r['sources'] = {str(original/p.relative_to(dst)): digest(p) for p in dst.rglob('*') if p.is_file()}
        helpers = [Path(__file__), Path(trace.__file__), Path(trace.paired.__file__),
                   Path(repeat_tool.__file__), trace.HERE/'trace.hpp']
        r['sources'].update({str(p): digest(p) for p in helpers})
        for rel in ('src/include', 'open_kernels', 'specs'):
            (out/'source'/rel).symlink_to(ROOT/rel, target_is_directory=True)
        cpp, cli = instrument((dst/'core.cpp').read_text(), (dst/'cli.cpp').read_text(), a.repeat)
        (dst/'core.cpp').write_text(cpp)
        (dst/'cli.cpp').write_text(cli)
        shutil.copy2(trace.HERE/'trace.hpp', dst/'trace.hpp')
        r['staged_sources'] = {str(p.relative_to(out)): digest(p) for p in dst.rglob('*') if p.is_file()}
        commands = [
            ['cmake', '-S', str(dst), '-B', str(out/'build'), '-DCMAKE_BUILD_TYPE=Release',
             f'-DXRT_INCLUDE_DIR={a.xrt_include.resolve()}', f'-DXRT_LIB_DIR={a.xrt_lib.resolve()}'],
            ['cmake', '--build', str(out/'build'), '-j', str(a.jobs)],
            ['ctest', '--test-dir', str(out/'build'), '--output-on-failure'],
        ]
        for cmd in commands:
            r['commands'].append(cmd)
            subprocess.run(cmd, check=True)
        deps = list((out/'build/CMakeFiles/open_qwen36_cli.dir').glob('*.o.d'))
        if len(deps) != 7:
            raise ValueError('incomplete compiler dependencies')
        files = set()
        for dep in deps:
            files.update(Path(p).resolve() for p in shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1]))
        r.update(compiler_dependencies={str(p): digest(p) for p in sorted(files)},
                 cli_sha256=digest(out/'build/open_qwen36_cli'),
                 cmake_cache_sha256=digest(out/'build/CMakeCache.txt'),
                 compiler=subprocess.check_output(['c++', '--version'], text=True))
        base.verify(r['sources'])
        r['complete'] = True
    except Exception as error:
        r['error'] = str(error)
        raise
    finally:
        (out/'build-evidence.json').write_text(json.dumps(r, indent=2)+'\n')


def verify(out, repeat):
    r = json.loads((out/'build-evidence.json').read_text())
    if not r['complete'] or r['repeat'] != repeat or not r['compiler_dependencies']:
        raise ValueError('incomplete or wrong capture build')
    base.verify(r['sources'])
    base.verify({str(out/p): h for p,h in r['staged_sources'].items()})
    base.verify(r['compiler_dependencies'])
    if digest(out/'build/open_qwen36_cli') != r['cli_sha256'] or digest(out/'build/CMakeCache.txt') != r['cmake_cache_sha256']:
        raise ValueError('capture executable or CMake cache changed')
    # Reconstruct the permitted instrumentation from the current shared engine.
    # No legacy paired/weighted patch or private kernel-loading logic is allowed.
    original = ROOT/'src/open_qwen36'
    cpp, cli = instrument((original/'core.cpp').read_text(), (original/'cli.cpp').read_text(), repeat)
    dst = out/'source/src/open_qwen36'
    if (dst/'core.cpp').read_text() != cpp or (dst/'cli.cpp').read_text() != cli:
        raise ValueError('unexpected engine instrumentation')
    for p in original.rglob('*'):
        if p.is_file() and p.suffix in ('.hpp', '.cpp') and p.name not in ('core.cpp', 'cli.cpp'):
            if p.read_bytes() != (dst/p.relative_to(original)).read_bytes():
                raise ValueError('shared arithmetic source changed: '+p.name)
    return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('out', 'xrt-include', 'xrt-lib'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--repeat', action='store_true')
    p.add_argument('--jobs', type=int, default=4)
    build(p.parse_args())
