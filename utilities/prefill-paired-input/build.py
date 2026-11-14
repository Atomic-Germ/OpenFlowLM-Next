"""Build private control/candidate CLIs without editing the production engine."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'private patch requires exactly one source anchor: {old!r}')
    return text.replace(old, new, 1)


def patch(cpp, hpp):
    if 'gemm_run_once' in cpp or 'paired_gemm_' in hpp:
        raise ValueError('source already contains the paired-input experiment')
    cpp = once(cpp, 'const float* Core::gemm_run(', 'const float* Core::gemm_run_once(')
    wrapper = '''const float* Core::gemm_run(const Step& s, const float* x, size_t T, size_t K, size_t N, int layer) {
    return paired_gemm_[s.args[2]].run(x, T * K, T * N, [&](const float* input) {
        return gemm_run_once(s, input, T, K, N, layer);
    });
}

'''
    cpp = once(cpp, 'void Core::gemm(', wrapper + 'void Core::gemm(')
    anchor = '    if (const char* env = std::getenv("OFLM_OPEN_STAGE_MAJOR")) stage_major_on_ = std::string(env) != "0";'
    cpp = once(cpp, anchor, anchor + '''
    if (stage_major_on_)
        throw std::runtime_error("stage-major is not supported by the paired-input experiment");''')
    decl = '    const float* gemm_run(const Step& s, const float* x, size_t T, size_t K, size_t N, int layer);'
    hpp = once(hpp, decl, decl + '\n' + decl.replace('gemm_run(', 'gemm_run_once(')
               + '\n    std::map<std::string, paired_gemm::Workspace> paired_gemm_;')
    hpp = once(hpp, '#include "stream_patch.hpp"',
               '#include "stream_patch.hpp"\n#include "open_qwen36/paired_gemm.hpp"')
    return cpp, hpp


def build(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)  # never overwrite a measured experiment
    evidence = dict(complete=False, variant=a.variant, sources={}, commands=[])
    try:
        stage = out / 'source'
        dst = stage / 'src/open_qwen36'
        sources = ROOT / 'src/open_qwen36'
        shutil.copytree(sources, dst, ignore=shutil.ignore_patterns('build', '__pycache__'))
        # The normal CMake include hierarchy remains intact; all core/CLI headers
        # come from the copy, so a changed Core layout cannot mix with the control.
        (stage / 'src/include').symlink_to(ROOT / 'src/include', target_is_directory=True)
        (stage / 'open_kernels').symlink_to(ROOT / 'open_kernels', target_is_directory=True)
        (stage / 'specs').symlink_to(ROOT / 'specs', target_is_directory=True)
        for p in dst.rglob('*'):
            if p.is_file():
                original = sources / p.relative_to(dst)
                evidence['sources'][str(original)] = digest(original)
        for name in ('build.py', 'paired_gemm.hpp'):
            evidence['sources'][str(HERE / name)] = digest(HERE / name)
        if a.variant == 'paired':
            cpp, hpp = patch((dst / 'core.cpp').read_text(), (dst / 'core.hpp').read_text())
            (dst / 'core.cpp').write_text(cpp)
            (dst / 'core.hpp').write_text(hpp)
            shutil.copy2(HERE / 'paired_gemm.hpp', dst / 'paired_gemm.hpp')
        evidence['staged_sources'] = {str(p.relative_to(out)): digest(p)
                                      for p in dst.rglob('*') if p.is_file()}
        commands = [
            ['cmake', '-S', str(dst), '-B', str(out / 'build'), '-DCMAKE_BUILD_TYPE=Release',
             f'-DXRT_INCLUDE_DIR={a.xrt_include.resolve()}', f'-DXRT_LIB_DIR={a.xrt_lib.resolve()}'],
            ['cmake', '--build', str(out / 'build'), '-j', str(a.jobs)],
            ['ctest', '--test-dir', str(out / 'build'), '--output-on-failure'],
        ]
        for cmd in commands:
            evidence['commands'].append(cmd)
            subprocess.run(cmd, check=True)
        evidence.update(complete=True, cli_sha256=digest(out / 'build/open_qwen36_cli'),
                        cmake_cache_sha256=digest(out / 'build/CMakeCache.txt'),
                        compiler=subprocess.check_output(['c++', '--version'], text=True))
    finally:
        (out / 'build-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=['control', 'paired'], required=True)
    p.add_argument('--xrt-include', type=Path, required=True)
    p.add_argument('--xrt-lib', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=4)
    build(p.parse_args())
