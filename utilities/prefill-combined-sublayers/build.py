"""Copy a verified combined CLI and add read-only linear/attention boundaries."""
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
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


runner=load('_combined_capture_runner',ROOT/'utilities/prefill-combined-cli/run.py')
prefix=load('_combined_capture_prefix',ROOT/'utilities/prefill-prefix-sublayers/build.py')
digest=runner.digest


def patch(cpp,cli):
    cpp,cli=prefix.patch(cpp,cli)
    old='#include "'+str(prefix.HERE/'capture.hpp')+'"'
    new='#include "'+str(HERE/'capture.hpp')+'"'
    once=runner.prep.builder.once
    cpp=once(cpp,old,new);cli=once(cli,old,new)
    start=cpp.index('void Core::block_layer_full(');end=cpp.index('void Core::gemm_phase(',start)
    body=cpp[start:end]
    anchor='    host::AttnGeom g;'
    captures=''.join(f'    sublayer_trace::rows("{name}", l, pos0, t_real, {width}, {name});\n'
                     for name,width in [('xn','hid'),('q','qw'),('k','kvw'),('v','kvw'),('gate','qw')])
    body=once(body,anchor,captures+anchor)
    anchor='    st.sync(XCL_BO_SYNC_BO_TO_DEVICE, t_real * row, pos0 * row);'
    body=once(body,anchor,'''    sublayer_trace::rows("og", l, pos0, t_real, qw, og);
    sublayer_trace::snapshot("kv_after", l, pos0, t_real, st.map<const uint8_t*>(), t_real * row);
'''+anchor)
    anchor='    const std::vector<float>& out = gout_;'
    body=once(body,anchor,anchor+'\n    sublayer_trace::row("proj", l, pos0, t_real, hid, out.data());')
    return cpp[:start]+body+cpp[end:],cli


def build(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    record=dict(complete=False,variant='combined-sublayer',native_validated=False,model_acceptance=False,commands=[])
    try:
        parent=a.parent.resolve();original=runner.prep.verify_build(parent,'combined')
        shutil.copytree(parent/'source',out/'source',symlinks=True)
        dst=out/'source/src/open_qwen36'
        cpp,cli=patch((dst/'core.cpp').read_text(),(dst/'cli.cpp').read_text())
        (dst/'core.cpp').write_text(cpp);(dst/'cli.cpp').write_text(cli)
        helpers=[Path(__file__),HERE/'capture.hpp',Path(prefix.__file__),Path(prefix.capture.__file__),
                 Path(prefix.capture.capture.__file__),Path(runner.__file__)]
        record.update(sources=original['sources']|{str(p):digest(p) for p in helpers},
                      kernel_inputs=original['kernel_inputs'],low_kernels=original['low_kernels'],
                      parent_build=str(parent/'build-evidence.json'),parent_build_sha256=digest(parent/'build-evidence.json'),
                      staged_sources={str(p.relative_to(out)):digest(p) for p in dst.rglob('*') if p.is_file()})
        commands=[['cmake','-S',str(dst),'-B',str(out/'build'),'-DCMAKE_BUILD_TYPE=Release',
                   f'-DXRT_INCLUDE_DIR={a.xrt_include.resolve()}',f'-DXRT_LIB_DIR={a.xrt_lib.resolve()}'],
                  ['cmake','--build',str(out/'build'),'-j',str(a.jobs)],
                  ['ctest','--test-dir',str(out/'build'),'--output-on-failure']]
        for command in commands:
            record['commands'].append(command);subprocess.run(command,check=True)
        deps=list((out/'build/CMakeFiles/open_qwen36_cli.dir').glob('*.o.d'))
        if len(deps)!=7:raise ValueError('missing compiler dependencies')
        files=set()
        for p in deps:files.update(Path(n).resolve() for n in shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
        runner.base.verify(record['sources'])
        record.update(complete=True,compiler_dependencies={str(p):digest(p) for p in sorted(files)},
                      cli_sha256=digest(out/'build/open_qwen36_cli'),cmake_cache_sha256=digest(out/'build/CMakeCache.txt'),
                      compiler=subprocess.check_output(['c++','--version'],text=True))
    except Exception as error:
        record['error']=str(error);raise
    finally:(out/'build-evidence.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','parent','xrt-include','xrt-lib'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4);build(p.parse_args())
