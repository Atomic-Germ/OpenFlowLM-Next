"""Build fresh paired-input, weighted-FFN and combined projection tracer CLIs."""
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
spec = importlib.util.spec_from_file_location('_combined_parent', ROOT/'utilities/prefill-weighted-cli/build.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
digest, once = parent.digest, parent.once
SHAPES = {'ffn': (34816, 5120), 'entry': (16384, 5120), 'out': (5120, 6144)}
DEFAULTS = {
    'ffn': ROOT/'Models/qwen38-27b/prefill-paired-weights/v2/low',
    'entry': ROOT/'Models/qwen38-27b/prefill-entry-replay/builds/low',
    'out': ROOT/'Models/qwen38-27b/prefill-out-replay/builds/low',
}


def verify_kernel(path, role):
    record = parent.kernels.verify_build(path, 'low')
    n, k = SHAPES[role]
    expected = dict(GQP_N=str(n), GQP_K=str(k), GQP_T='256', GQP_FMT='q4_1', GQP_BREUSE='0')
    if any(record['environment'].get(key) != value for key, value in expected.items()):
        raise ValueError('incompatible low-weight kernel geometry: '+role)
    return record


def patch(cpp, hpp, variant, lows):
    if variant == 'paired':
        return parent.paired.patch(cpp, hpp)
    if variant == 'weighted':
        return parent.patch(cpp, hpp, lows['ffn'])
    if variant != 'combined':
        raise ValueError('unknown private variant')
    cpp, hpp = parent.patch(cpp, hpp, lows['ffn'])
    start = cpp.index('    if (s.kernel != "gemm_n34816_k5120") return high();')
    end = cpp.index('    return weighted_gemm_', start)
    cpp = cpp[:start] + '''    const auto& gb = types_.at(layer)->gemm_block;
    const std::string low = combined::low_kernel(gb, s, T, K, N);
    if (low.empty()) return high();
    Step lo = s; lo.kernel = low;
''' + cpp[end:]
    anchor = '    logits_host_.assign(w_->vocab, 0.f);'
    setup = '''    for (int l = 0; l < nl_; ++l)
        combined::validate(types_[l]->gemm_block, w_->hidden);
'''
    for role in ('entry', 'out'):
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
    hpp = once(hpp, '#include "open_qwen36/weighted_gemm.hpp"',
               '#include "open_qwen36/weighted_gemm.hpp"\n#include "open_qwen36/policy.hpp"')
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
        roles = () if a.variant == 'paired' else ('ffn',) if a.variant == 'weighted' else tuple(SHAPES)
        for role in roles:
            record = verify_kernel(lows[role], role)
            evidence['kernel_inputs'].update({str(lows[role]/'build.json'): digest(lows[role]/'build.json')} | record['binaries'])
        evidence['low_kernels'] = {role: str(lows[role]) for role in roles}
        dst = out/'source/src/open_qwen36'
        original = ROOT/'src/open_qwen36'
        shutil.copytree(original, dst, ignore=shutil.ignore_patterns('build', '__pycache__'))
        evidence['sources'] = {str(original/p.relative_to(dst)): digest(p) for p in dst.rglob('*') if p.is_file()}
        headers = [HERE/'policy.hpp', parent.HERE/'weighted_gemm.hpp',
                   parent.trace.HERE/'trace.hpp', parent.paired.HERE/'paired_gemm.hpp']
        helpers = [Path(__file__), Path(parent.__file__), Path(parent.trace.__file__),
                   Path(parent.paired.__file__), Path(parent.kernels.__file__), *headers]
        evidence['sources'].update({str(p): digest(p) for p in helpers})
        for rel in ('src/include', 'open_kernels', 'specs'):
            (out/'source'/rel).symlink_to(ROOT/rel, target_is_directory=True)
        cpp, hpp = patch((dst/'core.cpp').read_text(), (dst/'core.hpp').read_text(), a.variant, lows)
        cpp, cli = parent.trace.patch(cpp, (dst/'cli.cpp').read_text())
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
            parent.kernels.verify(mapping)
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
    parser.add_argument('--variant', choices=['paired', 'weighted', 'combined'], required=True)
    for role, path in DEFAULTS.items():
        parser.add_argument('--low-'+role, type=Path, default=path)
    parser.add_argument('--jobs', type=int, default=4)
    build(parser.parse_args())
