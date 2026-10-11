"""Prepare or score an isolated norm/up/gate NPU probe from retained layer inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'open_kernels'),str(ROOT/'open_kernels/model')]
from decode_bundle import load_bundle
from decode_reference import load_reference,read_capture,sfx
from ffn_front import compare_front,read_front,metric
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from q4nx import Q4NX


def read_outputs(path,ff):
    a=read_capture(path,3*ff).reshape(ff//64,3,64)
    return tuple(a[:,i,:].reshape(-1) for i in range(3))


def score_arrays(res,nw,up,gate,x,outputs,eps):
    u,g,h=outputs
    front=compare_front(res,nw,up,gate,x,h,eps)
    ref=res.astype(np.float64)/np.sqrt(np.mean(res.astype(np.float64)**2)+eps)*nw
    ur=metric(u,up@ref);gr=metric(g,gate@ref)
    return dict(passed=front['passed'] and ur['passed'] and gr['passed'],front=front,up=ur,gate=gr)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def verify(out,expected_files=None):
    audit=json.loads((out/'audit.json').read_text())
    if expected_files is not None and set(audit['sha256'])!={str(p.resolve()) for p in expected_files}:
        raise ValueError('input hash coverage mismatch')
    for name,expected in audit['sha256'].items():
        if digest(name)!=expected:
            raise ValueError(f'input hash mismatch: {name}')
    return audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('model-dir','kernel-dir','fixture','pools','out','build'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--layer',type=int,default=11)
    ap.add_argument('--score',action='store_true')
    a=ap.parse_args()
    if a.out.resolve()==a.fixture.resolve():raise ValueError('separate output directory required')
    a.out.mkdir(parents=True,exist_ok=True)
    # Invalidate stale score before any verification or hardware preparation.
    (a.out/'score.json').unlink(missing_ok=True)
    spec=spec_from_model_dir(a.model_dir);meta=load_reference(a.fixture)
    m=load_bundle(a.kernel_dir,manifest(spec,max_ctx=meta['max_ctx']))
    if spec.family!='qwen35' or 'ffn' in spec.q8_roles or spec.hidden%1024:
        raise ValueError('requires qwen35 Q4 FFN, hidden divisible by 1024')
    if not 0<=a.layer<meta['layers'] or any(m[k]!=meta[k] for k in ('build_key','spec_hash')):
        raise ValueError('invalid layer or fixture identity')
    L=layout(spec,max_ctx=meta['max_ctx']);h=spec.hidden;ff=spec.intermediate
    pool=(a.pools/f'pool_L{a.layer}.bin').resolve();b=a.build.resolve();out=a.out.resolve()
    if pool.stat().st_size!=L.POOL_BYTES:raise ValueError('pool size mismatch')
    model=Q4NX(a.model_dir/'model.q4nx');pre=f'model.layers.{a.layer}.'
    nw=model.bf16(pre+'post_attention_layernorm.weight').astype(np.float64)
    inputs=[]
    for t in range(meta['tokens']):
        p=a.fixture/f'y_act{a.layer}{sfx(t)}.bin'
        r=read_front(p,L,h,ff,spec.layer_types[a.layer]=='full_attention',
                     norm_dtype=m['layout'].get('ffn_norm_dtype','bfloat16'))
        inputs.append((p,r['res']))
    cfg=['device',f'xclbin P {b}/final.xclbin',f'kernelx front P {b}/insts.bin',
         f'buf pool {L.POOL_BYTES} {pool}',f'buf rn {h*6}',f'buf norm {h*4}',f'buf out {ff*12}']
    for t in range(meta['tokens']):
        cfg += [f'load rn {out}/input_t{t}.bin','run front pool rn norm out',
                f'dump norm {out}/norm_t{t}.bin {h*4}',f'dump out {out}/front_t{t}.bin {ff*12}']
    cfg='\n'.join(cfg)+'\n'
    files=[out/'run.cfg',a.model_dir/'model.q4nx',pool,b/'final.xclbin',b/'insts.bin',a.fixture/'decode_reference.json',
               *[p for p,_ in inputs],*[out/f'input_t{t}.bin' for t in range(meta['tokens'])]]
    if not a.score:
        from q4nx import f32_to_bf16
        for t,(_,res) in enumerate(inputs):
            (out/f'input_t{t}.bin').write_bytes(res.astype(np.float32).tobytes()+f32_to_bf16(nw.astype(np.float32)).tobytes())
        (out/'run.cfg').write_text(cfg)
        audit=dict(layer=a.layer,tokens=meta['tokens'],spec_hash=m['spec_hash'],build_key=m['build_key'],
                   sha256={str(p.resolve()):digest(p) for p in files})
        (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(out/'run.cfg');return 0
    audit=verify(out,files)
    if (out/'run.cfg').read_text()!=cfg or audit['layer']!=a.layer or audit['tokens']!=meta['tokens']:
        raise ValueError('requested replay differs from pinned configuration')
    up=model.matmul_w(pre+'mlp.up_proj.weight',ff,h).astype(np.float64)
    gate=model.matmul_w(pre+'mlp.gate_proj.weight',ff,h).astype(np.float64)
    rows=[];captures={}
    for t,(_,res) in enumerate(inputs):
        x=read_capture(out/f'norm_t{t}.bin',h);outputs=read_outputs(out/f'front_t{t}.bin',ff)
        rows.append(dict(token=t,**score_arrays(res,nw,up,gate,x,outputs,spec.norm_eps)))
        for p in (out/f'norm_t{t}.bin',out/f'front_t{t}.bin'):
            captures[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    record=dict(passed=all(r['passed'] for r in rows),layer=a.layer,results=rows,
                scope='isolated FFN front from retained residual, not full-layer propagation',
                audit_sha256=hashlib.sha256((out/'audit.json').read_bytes()).hexdigest(),capture_sha256=captures)
    (out/'score.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2,allow_nan=False));return 0 if record['passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError) as e:
        print(f'FAIL: {e}',file=sys.stderr);sys.exit(1)
