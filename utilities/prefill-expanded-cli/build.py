"""Build exact combined control and expanded attention/down prefill tracer CLIs."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_expanded_parent', ROOT/'utilities/prefill-combined-cli/build.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
spec = importlib.util.spec_from_file_location('_expanded_repeat', ROOT/'utilities/prefill-combined-cli/repeat.py')
repeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repeat)
digest, once = parent.digest, parent.once
SHAPES = parent.SHAPES | {'attention': (14336, 5120), 'down': (5120, 17408)}
DEFAULTS = parent.DEFAULTS | {
    'attention': ROOT/'Models/qwen38-27b/prefill-attention-replay/builds/low',
    'down': ROOT/'Models/qwen38-27b/prefill-down-replay/builds/low',
}
VARIANTS = ('combined', 'expanded', 'expanded-repeat')


def roles_for(variant):
    if variant not in VARIANTS:
        raise ValueError('unknown private variant')
    return tuple(parent.SHAPES if variant == 'combined' else SHAPES)


def verify_kernel(path, role):
    record = parent.parent.kernels.verify_build(path, 'low')
    n, k = SHAPES[role]
    expected = dict(GQP_N=str(n), GQP_K=str(k), GQP_T='256', GQP_FMT='q4_1', GQP_BREUSE='0')
    if any(record['environment'].get(key) != value for key, value in expected.items()):
        raise ValueError('incompatible low-weight kernel geometry: '+role)
    return record


def patch(cpp, hpp, variant, lows):
    roles_for(variant)
    cpp, hpp = parent.patch(cpp, hpp, 'combined', lows)
    if variant == 'combined':
        return cpp, hpp
    cpp = once(cpp, 'combined::low_kernel', 'expanded::low_kernel')
    cpp = once(cpp, 'combined::validate(types_[l]->gemm_block, w_->hidden)',
               'expanded::validate(types_[l]->gemm_block, w_->hidden)')
    anchor = '    logits_host_.assign(w_->vocab, 0.f);'
    setup = '    expanded::validate_manifest(*w_);\n'
    for role in ('attention', 'down'):
        name = 'private_'+role+'_low'
        setup += '''    {
        xrt::xclbin low_xcl(std::string{XCL});
        auto uuid = dev_->register_xclbin(low_xcl);
        ctxs_[NAME] = std::make_unique<xrt::hw_context>(*dev_, uuid);
        KernelDesc desc; desc.context = NAME; desc.insts = INST;
        load_kernel(NAME, desc);
    }
'''.replace('XCL', json.dumps(str(lows[role]/'final.xclbin'))).replace(
            'INST', json.dumps(str(lows[role]/'insts.bin'))).replace('NAME', json.dumps(name))
    cpp = once(cpp, anchor, setup+anchor)
    hpp = once(hpp, '#include "open_qwen36/policy.hpp"',
               '#include "open_qwen36/expanded_policy.hpp"')
    return cpp, hpp


def build(a):
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    evidence = dict(complete=False, variant=a.variant, native_validated=False,
                    model_acceptance=False, sources={}, commands=[], kernel_inputs={})
    try:
        lows = {role: getattr(a, 'low_'+role, path).resolve() for role, path in DEFAULTS.items()}
        roles = roles_for(a.variant)
        for role in roles:
            record = verify_kernel(lows[role], role)
            evidence['kernel_inputs'].update({str(lows[role]/'build.json'): digest(lows[role]/'build.json')} | record['binaries'])
        evidence['low_kernels'] = {role: str(lows[role]) for role in roles}
        dst = out/'source/src/open_qwen36'
        original = ROOT/'src/open_qwen36'
        shutil.copytree(original, dst, ignore=shutil.ignore_patterns('build', '__pycache__'))
        evidence['sources'] = {str(original/p.relative_to(dst)): digest(p) for p in dst.rglob('*') if p.is_file()}
        headers = [HERE/'expanded_policy.hpp', parent.HERE/'policy.hpp', parent.parent.HERE/'weighted_gemm.hpp',
                   parent.parent.trace.HERE/'trace.hpp', parent.parent.paired.HERE/'paired_gemm.hpp']
        helpers = [Path(__file__), Path(parent.__file__), Path(parent.parent.__file__),
                   Path(parent.parent.trace.__file__), Path(parent.parent.paired.__file__),
                   Path(parent.parent.kernels.__file__), Path(repeat.__file__), *headers]
        evidence['sources'].update({str(p): digest(p) for p in helpers})
        for rel in ('src/include', 'open_kernels', 'specs'):
            (out/'source'/rel).symlink_to(ROOT/rel, target_is_directory=True)
        cpp, hpp = patch((dst/'core.cpp').read_text(), (dst/'core.hpp').read_text(), a.variant, lows)
        cpp, cli = parent.parent.trace.patch(cpp, (dst/'cli.cpp').read_text())
        if a.variant == 'expanded-repeat':
            cli = repeat.patch(cli)
        for name, text in [('core.cpp', cpp), ('core.hpp', hpp), ('cli.cpp', cli)]:
            (dst/name).write_text(text)
        for path in headers:
            shutil.copy2(path, dst/path.name)
        evidence['staged_sources'] = {str(p.relative_to(out)): digest(p) for p in dst.rglob('*') if p.is_file()}
        commands = [
            ['cmake', '-S', str(dst), '-B', str(out/'build'), '-DCMAKE_BUILD_TYPE=Release',
             f'-DXRT_INCLUDE_DIR={a.xrt_include.resolve()}', f'-DXRT_LIB_DIR={a.xrt_lib.resolve()}'],
            ['cmake', '--build', str(out/'build'), '-j', str(a.jobs)],
            ['ctest', '--test-dir', str(out/'build'), '--output-on-failure'],
        ]
        for command in commands:
            evidence['commands'].append(command)
            subprocess.run(command, check=True)
        deps = list((out/'build/CMakeFiles/open_qwen36_cli.dir').glob('*.o.d'))
        if len(deps) != 7:
            raise ValueError('incomplete compiler dependencies')
        paths = set()
        for dep in deps:
            paths.update(Path(p).resolve() for p in shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1]))
        for mapping in (evidence['sources'], {str(out/p): h for p, h in evidence['staged_sources'].items()}):
            parent.parent.kernels.verify(mapping)
        evidence.update(complete=True, compiler_dependencies={str(p): digest(p) for p in sorted(paths)},
                        cli_sha256=digest(out/'build/open_qwen36_cli'),
                        cmake_cache_sha256=digest(out/'build/CMakeCache.txt'),
                        compiler=subprocess.check_output(['c++', '--version'], text=True))
    except Exception as error:
        evidence['error'] = str(error)
        raise
    finally:
        (out/'build-evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'xrt-include', 'xrt-lib'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    for role, path in DEFAULTS.items():
        parser.add_argument('--low-'+role, type=Path, default=path)
    parser.add_argument('--jobs', type=int, default=4)
    build(parser.parse_args())
