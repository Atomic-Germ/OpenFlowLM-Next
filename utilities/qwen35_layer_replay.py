#!/usr/bin/env python3
"""Prepare an isolated layer replay from a completed Qwen3.5 capture.

Candidate kernels are recorded as experimental, never relabelled as an export.
Run the emitted cfg with run_kernel; use --score afterwards for the FP64 down gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'open_kernels'), str(ROOT/'open_kernels/model')]
from decode_bundle import load_bundle
from decode_reference import load_reference, read_capture, sfx
from ffn_partials import compare_partials, read_regions
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout, ffn_geometry
from q4nx import Q4NX


def replay_cfg(m, layer, tokens, fixture, pools, out, kernel):
    if not 0 <= layer < len(m['layers']) or tokens <= 0:
        raise ValueError('replay requires a covered layer and positive token count')
    kind = m['layers'][layer]
    if kind not in ('linear_attention', 'full_attention'):
        raise ValueError('unsupported layer type')
    full = kind == 'full_attention'
    kn = 'ax' if full else 'lx'
    b = m['layer_types'][kind]['buffers']
    hidden = m['layout']['hidden']
    args = ['pool', 'xres', 'consts', 'state', 'act'] + (['ptab'] if full else [])
    if m['layer_types'][kind]['program'] != [dict(op='run', kernel=kn, args=args)]:
        raise ValueError('unsupported layer program')
    lines = ['device', f'xclbin {kn} {kernel}/final.xclbin',
             f'kernelx {kn} {kn} {kernel}/insts.bin',
             f'buf pool {m["layout"]["pool_bytes"]} {pools}/pool_L{layer}.bin',
             f'buf consts {b["consts"]} {fixture}/consts_{layer}.bin',
             f'buf act {b["act"]}', f'buf xres {hidden*4}']
    if full:
        ctx = m['max_ctx_default']
        if tokens > ctx:
            raise ValueError('tokens exceed context capacity')
        lines += [f'buf state {b["state"]["row"]*ctx}',
                  f'buf ptab {m["layout"]["ptab_row"]*ctx} {fixture}/ptab.bin',
                  f'attngeom {m["layout"]["kv_row"]} {m["layout"]["ptab_row"]}']
    else:
        lines += [f'buf state {b["state"]["bytes"]} {fixture}/zstate_linear_attention.bin']
    for token in range(tokens):
        suffix = sfx(token)
        inp = fixture / (f'y_res{layer-1}{suffix}.bin' if layer else f'xres{token}.bin')
        lines += [f'load xres {inp}']
        if full:
            lines += [f'attnpos {kn} {token}']
        lines += [f'run {kn} ' + ' '.join(args),
                  f'dump act {out}/y_act{layer}{suffix}.bin {b["act"]}',
                  f'dump xres {out}/y_res{layer}{suffix}.bin {hidden*4}']
    return '\n'.join(lines)+'\n'


def verify_replay(out, cfg):
    if (out/'run.cfg').read_text() != cfg:
        raise ValueError('replay config differs from requested experiment')
    audit = json.loads((out/'replay.json').read_text())
    for name, expected in audit['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'replay input hash mismatch: {name}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','kernel-dir','fixture','pools','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--layer',type=int,default=60)
    ap.add_argument('--candidate',type=Path,help='experimental lx/ax build; never modifies the verified export')
    ap.add_argument('--score',action='store_true')
    a=ap.parse_args()
    spec=spec_from_model_dir(a.model_dir)
    if spec.family != 'qwen35': raise ValueError('qwen35 required')
    meta=load_reference(a.fixture)
    m=load_bundle(a.kernel_dir,manifest(spec,max_ctx=meta['max_ctx']))
    if any(meta[k] != m[k] for k in ('spec_hash','build_key')) or not 0 <= a.layer < meta['layers']:
        raise ValueError('fixture/export identity or layer coverage mismatch')
    full = spec.layer_types[a.layer] == 'full_attention'
    kernel=(a.candidate or a.kernel_dir/('ax' if full else 'lx')).resolve()
    cfg=replay_cfg(m,a.layer,meta['tokens'],a.fixture.resolve(),a.pools.resolve(),a.out.resolve(),kernel)
    a.out.mkdir(parents=True,exist_ok=True)
    if a.out.resolve()==a.fixture.resolve():raise ValueError('replay must not overwrite source captures')
    if a.score:
        verify_replay(a.out, cfg)
        offsets=layout(spec,max_ctx=meta['max_ctx']);split=tuple(ffn_geometry(spec).DOWN_SPLIT)
        model=Q4NX(a.model_dir/'model.q4nx')
        w=model.matmul_w(f'model.layers.{a.layer}.mlp.down_proj.weight',spec.hidden,spec.intermediate).astype(np.float64)
        results=[]
        for token in range(meta['tokens']):
            suffix=sfx(token)
            got=read_regions(a.out/f'y_act{a.layer}{suffix}.bin',offsets,spec.hidden,spec.intermediate,full)
            old=read_regions(a.fixture/f'y_act{a.layer}{suffix}.bin',offsets,spec.hidden,spec.intermediate,full)
            result=compare_partials(got['h'],w,got['res'],got['partials'],
                                   read_capture(a.out/f'y_res{a.layer}{suffix}.bin',spec.hidden),split)
            result.update(token=token,input_h_identical=got['h'].tobytes() == old['h'].tobytes(),
                          pre_ffn_residual_identical=got['res'].tobytes() == old['res'].tobytes())
            results.append(result)
        passed=all(r['passed'] and r['input_h_identical'] and r['pre_ffn_residual_identical'] for r in results)
        captures = [a.out/f'y_{kind}{a.layer}{sfx(t)}.bin'
                    for t in range(meta['tokens']) for kind in ('act', 'res')]
        record=dict(passed=passed,layer=a.layer,experimental=bool(a.candidate),results=results,
                    replay_sha256=hashlib.sha256((a.out/'replay.json').read_bytes()).hexdigest(),
                    capture_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in captures})
        (a.out/'score.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2));return 0 if passed else 1
    files=[kernel/'final.xclbin',kernel/'insts.bin',a.fixture/'decode_reference.json',
           a.fixture/f'consts_{a.layer}.bin',a.fixture/('ptab.bin' if full else 'zstate_linear_attention.bin'),a.pools/f'pool_L{a.layer}.bin']
    for token in range(meta['tokens']):
        inp=a.fixture/(f'y_res{a.layer-1}{sfx(token)}.bin' if a.layer else f'xres{token}.bin')
        read_capture(inp,spec.hidden);files.append(inp)
    audit=dict(experimental=bool(a.candidate),layer=a.layer,tokens=meta['tokens'],
               export_build_key=m['build_key'],sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (a.out/'replay.json').write_text(json.dumps(audit,indent=2)+'\n')
    (a.out/'run.cfg').write_text(cfg)
    print(a.out/'run.cfg');return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,KeyError) as e:
        print(f'FAIL: {e}',file=sys.stderr);sys.exit(1)
