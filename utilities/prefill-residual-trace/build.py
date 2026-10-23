"""Build a private paired-input CLI with synchronized per-layer residual capture."""
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
s = importlib.util.spec_from_file_location('_trace_paired_builder', ROOT / 'utilities/prefill-paired-input/build.py')
paired = importlib.util.module_from_spec(s)
s.loader.exec_module(paired)
once, digest = paired.once, paired.digest


def patch(cpp, cli):
    include = '#include "open_qwen36/core.hpp"'
    cpp = once(cpp, include, include + '\n#include "open_qwen36/trace.hpp"')
    cli = once(cli, include, include + '\n#include "open_qwen36/trace.hpp"')
    # A queued next layer could overwrite global xres before the host reads it.
    # Keep the existing serial dispatch body and wait for every layer.
    cpp = once(cpp, 'const bool pipe = submit_ahead_ > 0 && !deepstack;',
               'const bool pipe = false; // private residual trace: finish each layer before readback')
    anchor = '''            continue;
        }
        // (1) the previous layer's last dispatch'''
    cpp = once(cpp, anchor, '''            read_back(xres, w_->hidden * sizeof(float), 0);
            residual_trace::writer.append(l, static_cast<size_t>(pos_), 1, w_->hidden, xres.map<const float*>());
''' + anchor)
    for call, position, rows in (
        ('        block_layer_moe(l, T, t_real, xres.data());\n    }', 'pos_', 't_real'),
        ('        block_layer_moe(l, TOT, N, xres.data());\n    }', 'pos0', 'N'),
    ):
        cpp = once(cpp, call, call[:-5] +
                   f'        residual_trace::writer.append(l, static_cast<size_t>({position}), {rows}, hid, xres.data());\n    }}')
    anchor = '        auto t0 = std::chrono::steady_clock::now();\n        Core core(a.cfg);'
    cli = once(cli, anchor, '''        if (a.dump_prefix.empty() || a.decode_route != "npu" || !a.decode_schedule.empty() ||
            a.twice || a.repeat != 1 || a.at_position != 0 || a.bench || a.bench_decode ||
            a.bench_step || a.det_step || a.det_full || !a.bench_kernel.empty() ||
            a.max_tokens != 2 || !a.prefill_logits)
            throw std::runtime_error("residual trace requires one NPU request, position zero, --prefill-logits, --dump-logits and --max-tokens 2");
        open_qwen36::residual_trace::writer = open_qwen36::residual_trace::Writer(a.dump_prefix);
''' + anchor)
    cli = once(cli, '        core.load_weights();', '''        if (core.weights().hidden != 5120 || (core.num_layers() != 4 && core.num_layers() != 64))
            throw std::runtime_error("residual trace supports Qwen3.8 H5120 with 4 or 64 layers only");
        core.load_weights();''')
    anchor = '        std::vector<int> first = request(core, a);'
    cli = once(cli, anchor, anchor + '''
        if (core.position() != static_cast<int>(a.ids.size() + 1))
            throw std::runtime_error("incomplete residual trace request");
        open_qwen36::residual_trace::writer.finish(core.num_layers(), a.ids.size() + 1, core.weights().hidden);''')
    return cpp, cli


def build(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = dict(complete=False, variant='paired-residual-trace', sources={}, commands=[],
                    scheduling='serial layer completion before xres readback', native_validated=False)
    try:
        stage = out / 'source'
        dst = stage / 'src/open_qwen36'
        original = ROOT / 'src/open_qwen36'
        shutil.copytree(original, dst, ignore=shutil.ignore_patterns('build', '__pycache__'))
        for path in dst.rglob('*'):
            if path.is_file():
                source = original / path.relative_to(dst)
                evidence['sources'][str(source)] = digest(source)
        for relative in ('src/include', 'open_kernels', 'specs'):
            (stage / relative).symlink_to(ROOT / relative, target_is_directory=True)
        helpers = [Path(__file__), HERE / 'trace.hpp', Path(paired.__file__), paired.HERE / 'paired_gemm.hpp']
        evidence['sources'].update({str(p): digest(p) for p in helpers})
        cpp, hpp = paired.patch((dst/'core.cpp').read_text(), (dst/'core.hpp').read_text())
        cpp, cli = patch(cpp, (dst/'cli.cpp').read_text())
        for name, text in [('core.cpp', cpp), ('core.hpp', hpp), ('cli.cpp', cli)]:
            (dst/name).write_text(text)
        shutil.copy2(HERE/'trace.hpp', dst/'trace.hpp')
        shutil.copy2(paired.HERE/'paired_gemm.hpp', dst/'paired_gemm.hpp')
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
        # GCC depfiles cover shared include symlinks and SDK headers as well.
        dependencies = set()
        depfiles = list((out/'build/CMakeFiles/open_qwen36_cli.dir').glob('*.o.d'))
        if len(depfiles) != 7:
            raise ValueError('incomplete CLI compiler dependency files')
        for depfile in depfiles:
            contents = depfile.read_text().replace('\\\n', ' ')
            for name in shlex.split(contents.split(':', 1)[1]):
                dependencies.add(Path(name).resolve())
        evidence['compiler_dependencies'] = {str(p): digest(p) for p in sorted(dependencies)}
        evidence.update(complete=True, cli_sha256=digest(out/'build/open_qwen36_cli'),
                        cmake_cache_sha256=digest(out/'build/CMakeCache.txt'),
                        compiler=subprocess.check_output(['c++', '--version'], text=True))
    finally:
        (out/'build-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'xrt-include', 'xrt-lib'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--jobs', type=int, default=4)
    build(p.parse_args())
