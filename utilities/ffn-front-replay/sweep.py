"""Separate FFN projections over every layer, with byte-exact production norm/H.

Replays captured pre-FFN residuals using one streamed weight pool. Independent
FP64 norm/up/gate/activation gates supplement, not replace, full-depth decode.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from front_build import ROOT,HERE,digest,verify_build
sys.path.insert(0,str(ROOT/'open_kernels/model'))
from check import score_arrays,read_outputs
from decode_bundle import load_bundle
from decode_reference import load_reference,read_capture,sfx
from ffn_front import read_front
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from q4nx import Q4NX,f32_to_bf16


def separate_output(out,inputs):
    out=out.resolve()
    if any(out.is_relative_to(p.resolve()) or p.resolve().is_relative_to(out) for p in inputs):
        raise ValueError('separate replay output required')


def verify_original(current,original):
    original={str((ROOT/Path(p)).resolve()):v for p,v in original.items()}
    if not current or any(original.get(str(Path(p).resolve()))!=v for p,v in current.items()):
        raise ValueError('original provenance mismatch')


def validate_fixture(spec,meta,bundle):
    if (meta['layers']!=spec.num_layers or meta['tokens']<2
            or any(meta[k]!=bundle[k] for k in ('build_key','spec_hash'))
            or bundle['layout'].get('q4_activation_sum')!='quantized_fp32'
            or bundle['layout'].get('ffn_norm_dtype')!='float32'):
        raise ValueError('requires complete matching production fixture')


def pack_input(res,norm):
    res=np.asarray(res,dtype='<f4');norm=np.asarray(norm,dtype=np.float32)
    if (res.ndim!=1 or not res.size or norm.shape!=res.shape
            or not np.isfinite(res).all() or not np.isfinite(norm).all()):
        raise ValueError('invalid norm input')
    return res.tobytes()+f32_to_bf16(norm).tobytes()


def score_case(res,nw,up,gate,x,outputs,eps,production_norm,production_h):
    result=score_arrays(res,nw,up,gate,x,outputs,eps)
    identical=dict(norm=x.tobytes()==production_norm.tobytes(),h=outputs[2].tobytes()==production_h.tobytes())
    return dict(result,passed=result['passed'] and all(identical.values()),
                production_identical=all(identical.values()),identity=identical)


def config(spec,offsets,layers,tokens,pools,out,build):
    h=spec.hidden;ff=spec.intermediate
    lines=['device',f'xclbin P {build}/final.xclbin',f'kernelx front P {build}/insts.bin',
           f'buf pool {offsets.POOL_BYTES}',f'buf rn {h*6}',f'buf norm {h*4}',f'buf out {ff*12}']
    for l in layers:
        lines += [f'load pool {pools}/pool_L{l}.bin']
        for t in range(tokens):
            lines += [f'load rn {out}/input_L{l}_t{t}.bin','run front pool rn norm out',
                      f'dump norm {out}/norm_L{l}_t{t}.bin {h*4}',f'dump out {out}/front_L{l}_t{t}.bin {ff*12}']
    return '\n'.join(lines)+'\n'


def context(paths):
    spec=spec_from_model_dir(paths['model']);meta=load_reference(paths['fixture'])
    bundle=load_bundle(paths['kernels'],manifest(spec,max_ctx=meta['max_ctx']))
    validate_fixture(spec,meta,bundle);verify_build(paths['build'],spec)
    return spec,meta,bundle,layout(spec,meta['max_ctx'])


def files(paths,out,meta):
    result=[paths['model']/'model.q4nx',paths['model']/'config.json',paths['fixture']/'decode_reference.json',
            paths['original_inputs'],paths['original_captures'],out/'run.cfg']
    result += [paths['build']/n for n in ('build.json','final.xclbin','insts.bin')]
    result += [paths['kernels']/n for n in ('spec.json','manifest.json','toolchain.json')]
    result += list(HERE.glob('*.py'))
    result += [ROOT/'open_kernels/model'/n for n in ('q4nx.py','ffn_front.py','decode_reference.py','decode_bundle.py')]
    for l in range(meta['layers']):
        result += [paths['pools']/f'pool_L{l}.bin']
        for t in range(meta['tokens']):
            result += [paths['fixture']/f'y_act{l}{sfx(t)}.bin',out/f'input_L{l}_t{t}.bin']
    return result


def original_files(paths,meta):
    return [paths['model']/'model.q4nx']+[paths['pools']/f'pool_L{l}.bin' for l in range(meta['layers'])]


def check_provenance(paths,meta,hashes):
    original=json.loads(paths['original_inputs'].read_text())
    verify_original({str(p):hashes[str(p)] for p in original_files(paths,meta)},original)
    captures=json.loads(paths['original_captures'].read_text())['sha256']['captures']
    current={str(p):hashes[str(p)] for l in range(meta['layers']) for t in range(meta['tokens'])
             for p in [paths['fixture']/f'y_act{l}{sfx(t)}.bin']}
    verify_original(current,captures)


def prepare(paths,out):
    separate_output(out,list(paths.values()))
    out.mkdir(parents=True,exist_ok=True)
    for n in ('score.json','audit.json','progress.json'):(out/n).unlink(missing_ok=True)
    spec,meta,bundle,off=context(paths);model=Q4NX(paths['model']/'model.q4nx')
    for l in range(meta['layers']):
        if (paths['pools']/f'pool_L{l}.bin').stat().st_size!=off.POOL_BYTES:raise ValueError('pool size mismatch')
        nw=model.bf16(f'model.layers.{l}.post_attention_layernorm.weight')
        for t in range(meta['tokens']):
            r=read_front(paths['fixture']/f'y_act{l}{sfx(t)}.bin',off,spec.hidden,spec.intermediate,
                         spec.layer_types[l]=='full_attention',norm_dtype='float32')
            (out/f'input_L{l}_t{t}.bin').write_bytes(pack_input(r['res'],nw))
            for kind in ('norm','front'):(out/f'{kind}_L{l}_t{t}.bin').unlink(missing_ok=True)
    (out/'run.cfg').write_text(config(spec,off,range(meta['layers']),meta['tokens'],paths['pools'],out,paths['build']))
    hashes={str(p.resolve()):digest(p) for p in files(paths,out,meta)}
    check_provenance(paths,meta,hashes)
    audit=dict(paths={k:str(v) for k,v in paths.items()},layers=meta['layers'],tokens=meta['tokens'],
               build_key=bundle['build_key'],sha256=hashes)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Prepared',meta['layers']*meta['tokens'],'FFN projection cases',flush=True)


def verified(out):
    a=json.loads((out/'audit.json').read_text());paths={k:Path(v) for k,v in a['paths'].items()}
    separate_output(out,list(paths.values()));spec,meta,bundle,off=context(paths)
    if a['layers']!=meta['layers'] or a['tokens']!=meta['tokens'] or a['build_key']!=bundle['build_key']:
        raise ValueError('replay coverage mismatch')
    expected={str(p.resolve()) for p in files(paths,out,meta)}
    if set(a['sha256'])!=expected:raise ValueError('input hash coverage mismatch')
    for p,h in a['sha256'].items():
        if digest(p)!=h:raise ValueError('input hash mismatch: '+p)
    check_provenance(paths,meta,a['sha256'])
    if (out/'run.cfg').read_text()!=config(spec,off,range(meta['layers']),meta['tokens'],paths['pools'],out,paths['build']):
        raise ValueError('replay configuration mismatch')
    return a,paths,spec,meta,off


def score(out):
    (out/'score.json').unlink(missing_ok=True)
    (out/'progress.json').unlink(missing_ok=True)
    a,paths,spec,meta,off=verified(out);model=Q4NX(paths['model']/'model.q4nx')
    rows=[];hashes={}
    for l in range(meta['layers']):
        pre=f'model.layers.{l}.'
        nw=model.bf16(pre+'post_attention_layernorm.weight').astype(np.float64)
        up=model.matmul_w(pre+'mlp.up_proj.weight',spec.intermediate,spec.hidden).astype(np.float64)
        gate=model.matmul_w(pre+'mlp.gate_proj.weight',spec.intermediate,spec.hidden).astype(np.float64)
        for t in range(meta['tokens']):
            r=read_front(paths['fixture']/f'y_act{l}{sfx(t)}.bin',off,spec.hidden,spec.intermediate,
                         spec.layer_types[l]=='full_attention',norm_dtype='float32')
            npth=out/f'norm_L{l}_t{t}.bin';fp=out/f'front_L{l}_t{t}.bin'
            x=read_capture(npth,spec.hidden);outputs=read_outputs(fp,spec.intermediate)
            row=score_case(r['res'],nw,up,gate,x,outputs,spec.norm_eps,r['xm'],r['h'])
            rows.append(dict(layer=l,token=t,**row))
            for p in (npth,fp):hashes[p.name]=digest(p)
        del up,gate
        (out/'progress.json').write_text(json.dumps(dict(completed=l+1,total=meta['layers'],passed=all(r['passed'] for r in rows)))+'\n')
        print('layer',l,'PASS' if all(r['passed'] for r in rows[-meta['tokens']:]) else 'FAIL',flush=True)
    result=dict(passed=all(r['passed'] for r in rows),scope=__doc__,layers=meta['layers'],tokens=meta['tokens'],
                build_key=a['build_key'],audit_sha256=digest(out/'audit.json'),capture_sha256=hashes,results=rows)
    (out/'score.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--score',action='store_true')
    for n in ('model','kernels','fixture','pools','build','original-inputs','original-captures'):p.add_argument('--'+n,type=Path)
    a=p.parse_args();out=a.out.resolve()
    if a.score:return 0 if score(out)['passed'] else 1
    names=('model','kernels','fixture','pools','build','original_inputs','original_captures')
    if any(getattr(a,n) is None for n in names):p.error('preparation requires every input path')
    prepare({n:getattr(a,n).resolve() for n in names},out);return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError) as e:
        print('FAIL:',e,file=sys.stderr);sys.exit(1)
