"""Build and validate two traced requests on one resident combined engine."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

spec=importlib.util.spec_from_file_location('_combined_resident',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
prep,builder,base,digest=runner.prep,runner.prep.builder,runner.base,runner.digest


def patch(cli):
    once=builder.once
    cli=once(cli,'    Args a = parse(argc, argv);',
             '    std::setvbuf(stdout, nullptr, _IONBF, 0);\n    Args a = parse(argc, argv);')
    cli=once(cli,'            a.twice || a.repeat != 1 ||','            !a.twice || a.repeat != 1 ||')
    cli=once(cli,'residual trace requires one NPU request,','residual trace requires --twice,')
    cli=once(cli,'        std::vector<int> first = request(core, a);',
             '        std::fprintf(stderr, "TRACE_REQUEST_BEGIN 0\\n");\n        std::vector<int> first = request(core, a);')
    start=cli.index('        int reps = a.twice ? 2 : a.repeat;')
    end=cli.index('        std::printf("DONE\\n");',start)
    cli=cli[:start]+'''        std::fprintf(stderr, "TRACE_REQUEST_END 0\\n");
        auto snap = core.checkpoint();
        core.restore(snap);
        Args again_args = a;
        again_args.dump_prefix = a.dump_prefix + "_repeat";
        if (!a.dump_act.empty()) {
            const auto colon = a.dump_act.rfind(':');
            if (colon == std::string::npos) throw std::runtime_error("invalid act capture specification");
            again_args.dump_act = a.dump_act.substr(0, colon + 1) + again_args.dump_prefix + "-act.bin";
        }
        open_qwen36::residual_trace::writer = open_qwen36::residual_trace::Writer(again_args.dump_prefix);
        std::fprintf(stderr, "TRACE_REQUEST_BEGIN 1\\n");
        std::vector<int> again = request(core, again_args); // request() calls Core::reset()
        if (core.position() != static_cast<int>(a.ids.size() + 1))
            throw std::runtime_error("incomplete repeated residual trace request");
        open_qwen36::residual_trace::writer.finish(core.num_layers(), a.ids.size() + 1, core.weights().hidden);
        if (!again_args.dump_act.empty()) dump_act_slice(core, again_args.dump_act);
        std::fprintf(stderr, "TRACE_REQUEST_END 1\\n");
        bool same = first == again;
        std::fprintf(stderr, "request 2 %s the first\\n", same ? "REPRODUCED" : "DIFFERS FROM");
        if (!same) return 1;
'''+cli[end:]
    return cli


def sections(log):
    markers=re.findall(r'^TRACE_REQUEST_(BEGIN|END) ([01])$',log,re.M)
    if markers != [('BEGIN','0'),('END','0'),('BEGIN','1'),('END','1')]:
        raise ValueError('incomplete or unordered repeated request log')
    if log.count('request 2 REPRODUCED the first')!=1 or not log.rstrip().endswith('DONE'):
        raise ValueError('repeated request did not reproduce tokens')
    return [log.split(f'TRACE_REQUEST_BEGIN {i}\n',1)[1].split(f'TRACE_REQUEST_END {i}\n',1)[0] for i in range(2)]


def build(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    record=dict(complete=False,variant='combined-repeat',native_validated=False,model_acceptance=False,commands=[])
    try:
        parent=a.parent.resolve();original=prep.verify_build(parent,'combined')
        shutil.copytree(parent/'source',out/'source',symlinks=True)
        dst=out/'source/src/open_qwen36'
        (dst/'cli.cpp').write_text(patch((dst/'cli.cpp').read_text()))
        record.update(sources=original['sources']|{str(Path(p).resolve()):digest(p) for p in
                                                 (Path(__file__),Path(runner.__file__),Path(prep.__file__))},
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
        for path in deps:
            files.update(Path(p).resolve() for p in shlex.split(path.read_text().replace('\\\n',' ').split(':',1)[1]))
        base.verify(record['sources'])
        record.update(complete=True,compiler_dependencies={str(p):digest(p) for p in sorted(files)},
                      cli_sha256=digest(out/'build/open_qwen36_cli'),cmake_cache_sha256=digest(out/'build/CMakeCache.txt'),
                      compiler=subprocess.check_output(['c++','--version'],text=True))
    except Exception as error:
        record['error']=str(error);raise
    finally:(out/'build-evidence.json').write_text(json.dumps(record,indent=2)+'\n')


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(complete=False,passed=False,model_acceptance=False,routes=[])
    try:
        baseline=a.baseline.resolve()
        original=json.loads((baseline/'report.json').read_text())
        audit=json.loads((baseline/'audit.json').read_text())
        if not original['complete'] or not audit['complete'] or audit['report_sha256']!=digest(baseline/'report.json'):
            raise ValueError('single-request run has not been independently audited')
        prepared=Path(original['prepared']);record,meta,offsets=runner.preparation(prepared)
        build_path=a.build.resolve();b=prep.verify_build(build_path,'combined-repeat')
        if digest(b['parent_build'])!=b['parent_build_sha256'] or Path(b['parent_build']).parent!=Path(record['builds']['combined']['directory']):
            raise ValueError('repeat build does not derive from the measured candidate')
        refs=prepared/'references';(out/'ids.txt').write_bytes((refs/'ids.txt').read_bytes())
        env={k:v for k,v in os.environ.items() if not k.startswith('OFLM_OPEN_')}
        env.update(OMP_NUM_THREADS='2',OMP_WAIT_POLICY='PASSIVE',OFLM_OPEN_ATTN_BLOCK='0',
                   OFLM_OPEN_LAYER_MAJOR='1',OFLM_OPEN_STAGE_MAJOR='0')
        with (prep.ROOT/'Models/qwen38-27b/.resident-validation.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for route,extra in runner.ROUTES.items():
                base.memory_preflight(meta['layers']);runner.device_available()
                command=base.native_command(build_path/'build/open_qwen36_cli',out,meta,offsets,route,extra+['--twice'])
                raw=out/(route+'-requests.log')
                print('repeated resident requests:',route,flush=True)
                with raw.open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
                parts=sections(raw.read_text())
                for prefix,text in zip((route,route+'_repeat'),parts):(out/(prefix+'.log')).write_text(text)
                first=runner.score_route(out,refs,meta,offsets,route)
                second=runner.score_route(out,refs,meta,offsets,route+'_repeat')
                identity=runner.require_identity(baseline/'combined',out,len(meta['ids']),meta['layers'],(route,))
                repeat_identity={name:digest(out/name)==digest(out/name.replace(route,route+'_repeat',1))
                                 for name in runner.capture_names(route,len(meta['ids']),meta['layers'])}
                if not all(repeat_identity.values()):raise ValueError('repeated captures differ')
                result['routes'].append(dict(route=route,first=first,second=second,baseline_identity=identity,
                                              repeat_identity=repeat_identity,command=command,raw_log_sha256=digest(raw)))
                print(route,'all',len(identity),'captures exact for first and repeated request',flush=True)
        result.update(complete=True,passed=True,numerical_passed=all(r['second']['passed'] for r in result['routes']),
                      cli_sha256=b['cli_sha256'],build=str(build_path),baseline=str(baseline),
                      baseline_report_sha256=digest(baseline/'report.json'),
                      preparation_sha256=digest(prepared/'preparation.json'),runner_sha256=digest(Path(__file__)),
                      scope='two identical requests per resident process, normal Core::reset; not full-buffer zeroing')
    except Exception as error:
        result.update(complete=False,passed=False,error=str(error));raise
    finally:(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['passed'] else 1


def audit(a):
    out=a.out.resolve();result=dict(complete=False,passed=False,model_acceptance=False)
    try:
        saved=json.loads((out/'report.json').read_text())
        if not saved['complete'] or saved['runner_sha256']!=digest(Path(__file__)):
            raise ValueError('incomplete repeat run or changed runner')
        baseline=Path(saved['baseline']);original=json.loads((baseline/'report.json').read_text())
        if digest(baseline/'report.json')!=saved['baseline_report_sha256']:
            raise ValueError('single-request baseline changed')
        prepared=Path(original['prepared']);record,meta,offsets=runner.preparation(prepared)
        if digest(prepared/'preparation.json')!=saved['preparation_sha256']:
            raise ValueError('preparation changed')
        b=prep.verify_build(Path(saved['build']),'combined-repeat')
        if b['cli_sha256']!=saved['cli_sha256'] or digest(b['parent_build'])!=b['parent_build_sha256']:
            raise ValueError('repeat build changed')
        if [r['route'] for r in saved['routes']]!=list(runner.ROUTES):
            raise ValueError('incomplete repeated routes')
        for row in saved['routes']:
            route=row['route'];raw=out/(route+'-requests.log')
            if digest(raw)!=row['raw_log_sha256']:raise ValueError('raw request log changed')
            parts=sections(raw.read_text())
            for key,prefix,text in zip(('first','second'),(route,route+'_repeat'),parts):
                if (out/(prefix+'.log')).read_text()!=text:raise ValueError('request log extraction changed')
                if runner.score_route(out,prepared/'references',meta,offsets,prefix)!=row[key]:
                    raise ValueError('reported repeat measurements differ')
            identity=runner.require_identity(baseline/'combined',out,len(meta['ids']),meta['layers'],(route,))
            repeated={name:digest(out/name)==digest(out/name.replace(route,route+'_repeat',1))
                      for name in runner.capture_names(route,len(meta['ids']),meta['layers'])}
            if identity!=row['baseline_identity'] or repeated!=row['repeat_identity'] or not all(repeated.values()):
                raise ValueError('repeat identity differs')
        numerical=all(r['second']['passed'] for r in saved['routes'])
        if not saved['passed'] or numerical!=saved['numerical_passed']:
            raise ValueError('incorrect repeat aggregate result')
        result.update(complete=True,passed=True,numerical_passed=numerical,report_sha256=digest(out/'report.json'))
    except Exception as error:
        result['error']=str(error);raise
    finally:(out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build')
    for name in ('out','parent','xrt-include','xrt-lib'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    p=sub.add_parser('run')
    for name in ('out','build','baseline'):p.add_argument('--'+name,type=Path,required=True)
    sub.add_parser('audit').add_argument('--out',type=Path,required=True)
    a=parser.parse_args();raise SystemExit({'build':build,'run':run,'audit':audit}[a.command](a))
