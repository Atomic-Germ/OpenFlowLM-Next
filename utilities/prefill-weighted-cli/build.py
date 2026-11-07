"""Build private paired-input control and paired-weight FFN causal tracer CLIs."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
trace=load('_weighted_trace',ROOT/'utilities/prefill-residual-trace/build.py')
paired=trace.paired
kernels=load('_weighted_kernel',ROOT/'utilities/prefill-paired-weights/run.py')
once,digest=paired.once,paired.digest

def patch(cpp,hpp,low):
    cpp,hpp=paired.patch(cpp,hpp)
    start=cpp.index('const float* Core::gemm_run(');end=cpp.index('\nvoid Core::gemm(',start)
    wrapper='''const float* Core::gemm_run(const Step& s, const float* x, size_t T, size_t K, size_t N, int layer) {
    auto high = [&]() {
        return paired_gemm_[s.args[2]].run(x, T * K, T * N, [&](const float* input) {
            return gemm_run_once(s, input, T, K, N, layer);
        });
    };
    if (s.kernel != "gemm_n34816_k5120") return high();
    const auto& gb = types_.at(layer)->gemm_block;
    if (T != 256 || K != 5120 || N != 34816 || s.split || gb.ff != 17408 ||
        gb.ffn_program.size() != 2 || gb.ffn_program[0].args != s.args)
        throw std::runtime_error("paired-weight experiment requires validated dense FFN up/gate");
    Step lo = s; lo.kernel = "private_ffn_low";
    return weighted_gemm_[s.args[2]].run(T * N, high, [&]() {
        return low_weight_gemm_[s.args[2]].run(x, T * K, T * N, [&](const float* input) {
            return gemm_run_once(lo, input, T, K, N, layer);
        });
    });
}
'''
    cpp=cpp[:start]+wrapper+cpp[end:]
    anchor='    logits_host_.assign(w_->vocab, 0.f);'
    setup='''    if (w_->hidden != 5120 || !wanted.count("gemm_n34816_k5120"))
        throw std::runtime_error("paired-weight CLI requires the validated Qwen3.8 wide export");
    {
        xrt::xclbin low_xcl(std::string{LOW_XCL});
        auto uuid = dev_->register_xclbin(low_xcl);
        ctxs_["private_ffn_low"] = std::make_unique<xrt::hw_context>(*dev_, uuid);
        KernelDesc desc; desc.context = "private_ffn_low"; desc.insts = LOW_INST;
        load_kernel("private_ffn_low", desc);
    }
'''.replace('LOW_XCL',json.dumps(str(low/'final.xclbin'))).replace('LOW_INST',json.dumps(str(low/'insts.bin')))
    cpp=once(cpp,anchor,setup+anchor)
    hpp=once(hpp,'    std::map<std::string, paired_gemm::Workspace> paired_gemm_;',
        '    std::map<std::string, paired_gemm::Workspace> paired_gemm_, low_weight_gemm_;\n'
        '    std::map<std::string, weighted_gemm::Workspace> weighted_gemm_;')
    hpp=once(hpp,'#include "open_qwen36/paired_gemm.hpp"',
        '#include "open_qwen36/paired_gemm.hpp"\n#include "open_qwen36/weighted_gemm.hpp"')
    return cpp,hpp

def build(a):
    out=a.out.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);e=dict(complete=False,variant=a.variant,commands=[],sources={})
    try:
        low=a.low.resolve();kb=kernels.verify_build(low,'low')
        e['kernel_inputs']={str(low/'build.json'):digest(low/'build.json')}|kb['binaries']
        dst=out/'source/src/open_qwen36';original=ROOT/'src/open_qwen36'
        shutil.copytree(original,dst,ignore=shutil.ignore_patterns('build','__pycache__'))
        e['sources']={str(original/p.relative_to(dst)):digest(p) for p in dst.rglob('*') if p.is_file()}
        helpers=[Path(__file__),HERE/'weighted_gemm.hpp',Path(trace.__file__),trace.HERE/'trace.hpp',
                 Path(paired.__file__),paired.HERE/'paired_gemm.hpp',Path(kernels.__file__)]
        e['sources'].update({str(p):digest(p) for p in helpers})
        for rel in ('src/include','open_kernels','specs'):
            (out/'source'/rel).symlink_to(ROOT/rel,target_is_directory=True)
        cpp,hpp=(dst/'core.cpp').read_text(),(dst/'core.hpp').read_text()
        cpp,hpp=patch(cpp,hpp,low) if a.variant=='candidate' else paired.patch(cpp,hpp)
        cpp,cli=trace.patch(cpp,(dst/'cli.cpp').read_text())
        for name,text in [('core.cpp',cpp),('core.hpp',hpp),('cli.cpp',cli)]: (dst/name).write_text(text)
        for path in (HERE/'weighted_gemm.hpp',trace.HERE/'trace.hpp',paired.HERE/'paired_gemm.hpp'):
            shutil.copy2(path,dst/path.name)
        e['staged_sources']={str(p.relative_to(out)):digest(p) for p in dst.rglob('*') if p.is_file()}
        commands=[['cmake','-S',str(dst),'-B',str(out/'build'),'-DCMAKE_BUILD_TYPE=Release',
                   f'-DXRT_INCLUDE_DIR={a.xrt_include.resolve()}',f'-DXRT_LIB_DIR={a.xrt_lib.resolve()}'],
                  ['cmake','--build',str(out/'build'),'-j',str(a.jobs)],
                  ['ctest','--test-dir',str(out/'build'),'--output-on-failure']]
        for cmd in commands:e['commands'].append(cmd);subprocess.run(cmd,check=True)
        deps=list((out/'build/CMakeFiles/open_qwen36_cli.dir').glob('*.o.d'))
        if len(deps)!=7:raise ValueError('incomplete compiler dependencies')
        paths=set()
        for dep in deps:
            paths.update(Path(p).resolve() for p in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]))
        e.update(complete=True,compiler_dependencies={str(p):digest(p) for p in sorted(paths)},
                 cli_sha256=digest(out/'build/open_qwen36_cli'),cmake_cache_sha256=digest(out/'build/CMakeCache.txt'),
                 compiler=subprocess.check_output(['c++','--version'],text=True))
    finally:(out/'build-evidence.json').write_text(json.dumps(e,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','xrt-include','xrt-lib'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['control','candidate'],required=True)
    p.add_argument('--low',type=Path,default=ROOT/'Models/qwen38-27b/prefill-paired-weights/v2/low')
    p.add_argument('--jobs',type=int,default=4);build(p.parse_args())
