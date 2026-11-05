"""Prepare/score full-ax entry replay without relabelling the production export."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
from entry_common import ROOT,HERE,geometry,source_hash,digest,read_candidate_norm
from attention_projections import score_layer,read_boundaries
from attention_heads import read_attention
from decode_reference import load_reference,read_capture,sfx
from decode_bundle import load_bundle
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.spec import FULL
from recipes.cache import build_key
from q4nx import Q4NX


def verify_inputs(audit,files):
    if set(audit['sha256'])!={str(p.resolve()) for p in files}:raise ValueError('input hash coverage mismatch')
    for name,h in audit['sha256'].items():
        if digest(name)!=h:raise ValueError(f'input hash mismatch: {name}')


def replay_cfg(m,g,layers,tokens,fixture,pools,out,build):
    l=m['layout'];b=m['layer_types']['full_attention']['buffers'];kv=b['state']['row']*m['max_ctx_default']
    cfg=['device',f'attngeom {l["kv_row"]} {l["ptab_row"]}',
         f'xclbin ax {build}/final.xclbin',f'kernelx ax ax {build}/insts.bin',
         f'buf pool {l["pool_bytes"]}',f'buf consts {b["consts"]}',f'buf xres {l["hidden"]*4}',
         f'buf act {g["act_bytes"]}',f'buf state {kv}',
         f'buf ptab {m["max_ctx_default"]*l["ptab_row"]} {fixture}/ptab.bin']
    for layer in layers:
        cfg += [f'load pool {pools}/pool_L{layer}.bin',f'load consts {fixture}/consts_{layer}.bin',
                f'load state {out}/zero.bin',f'load act {out}/zero_act.bin']
        for t in range(tokens):
            suffix=sfx(t);inp=fixture/(f'y_res{layer-1}{suffix}.bin' if layer else f'xres{t}.bin')
            cfg += [f'load xres {inp}',f'attnpos ax {t}','run ax pool xres consts state act ptab',
                    f'dump act {out}/y_act{layer}{suffix}.bin {g["act_bytes"]}',
                    f'dump xres {out}/y_res{layer}{suffix}.bin {l["hidden"]*4}']
        cfg += [f'dump state {out}/kv_L{layer}.bin {tokens*b["state"]["row"]}']
    return '\n'.join(cfg)+'\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('model-dir','kernel-dir','fixture','pools','build','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--layers',help='default: all full-attention layers')
    ap.add_argument('--score',action='store_true');a=ap.parse_args()
    out=a.out.resolve();fixture=a.fixture.resolve();build=a.build.resolve();pools=a.pools.resolve()
    if out==fixture or out.is_relative_to(a.kernel_dir.resolve()):raise ValueError('separate experimental output required')
    out.mkdir(parents=True,exist_ok=True);(out/'score.json').unlink(missing_ok=True)
    spec=spec_from_model_dir(a.model_dir);meta=load_reference(fixture)
    m=load_bundle(a.kernel_dir,manifest(spec,max_ctx=meta['max_ctx']))
    if any(meta[k]!=m[k] for k in ('spec_hash','build_key')):raise ValueError('fixture identity mismatch')
    if meta['hidden']!=spec.hidden or not 0<meta['layers']<=spec.num_layers:raise ValueError('fixture dimensions mismatch')
    bm=json.loads((build/'build.json').read_text());mode=bm['mode'];g=geometry(spec,mode)
    if (bm['geometry']!=g or bm['production_build_key']!=build_key(spec) or
            bm['production_build_key']!=m['build_key'] or bm['source_hash']!=source_hash(spec,mode)):
        raise ValueError('build source/geometry mismatch')
    for n in ('final.xclbin','insts.bin'):
        if digest(build/n)!=bm['sha256'][n]:raise ValueError('build binary hash mismatch')
    if mode=='bf16' and any(digest(build/n)!=digest(a.kernel_dir/'ax'/n) for n in ('final.xclbin','insts.bin')):
        raise ValueError('control differs from production binaries')
    if mode=='fp32' and digest(build/'entry_ax.py')!=bm['design_sha256']:raise ValueError('generated design hash mismatch')
    covered=[l for l in range(meta['layers']) if spec.layer_types[l]==FULL]
    layers=covered if a.layers is None else [int(v) for v in a.layers.split(',')]
    if not layers or len(set(layers))!=len(layers) or any(l not in covered for l in layers):raise ValueError('invalid layer coverage')
    cfg=replay_cfg(m,g,layers,meta['tokens'],fixture,pools,out,build)
    files=[build/n for n in ('build.json','final.xclbin','insts.bin')]+[fixture/'decode_reference.json',fixture/'ptab.bin',
           a.model_dir/'model.q4nx',a.model_dir/'config.json',out/'run.cfg',out/'zero.bin',out/'zero_act.bin',Path(__file__),
           ROOT/'open_kernels/model/attention_projections.py',ROOT/'open_kernels/model/attention_heads.py',
           ROOT/'open_kernels/model/ffn_front.py',ROOT/'open_kernels/model/q4nx.py']
    for layer in layers:
        files += [fixture/f'consts_{layer}.bin',pools/f'pool_L{layer}.bin']
        for t in range(meta['tokens']):
            suffix=sfx(t)
            inp=fixture/(f'y_res{layer-1}{suffix}.bin' if layer else f'xres{t}.bin')
            read_capture(inp,spec.hidden)
            files += [inp,fixture/f'y_act{layer}{suffix}.bin',fixture/f'y_res{layer}{suffix}.bin']
    identity=dict(mode=mode,layers=layers,tokens=meta['tokens'],geometry=g,build_source_hash=bm['source_hash'],
                  fixture_build_key=m['build_key'])
    if not a.score:
        (out/'run.cfg').write_text(cfg)
        (out/'zero_act.bin').write_bytes(bytes(g['act_bytes']))
        (out/'zero.bin').write_bytes(bytes(meta['max_ctx']*m['layout']['kv_row']))
        audit=dict(identity,sha256={str(p.resolve()):digest(p) for p in files})
        (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(out/'run.cfg');return 0
    audit=json.loads((out/'audit.json').read_text());verify_inputs(audit,files)
    if any(audit[k]!=v for k,v in identity.items()) or (out/'run.cfg').read_text()!=cfg:raise ValueError('replay config mismatch')
    L=layout(spec,max_ctx=meta['max_ctx']);E=replace(L,AA_BYTES=g['act_bytes'])
    model=Q4NX(a.model_dir/'model.q4nx');rows=[];hashes={};weight_hashes={};control=True
    for layer in layers:
        captures=[];inputs=[]
        for t in range(meta['tokens']):
            suffix=sfx(t);p=out/f'y_act{layer}{suffix}.bin'
            inp=fixture/(f'y_res{layer-1}{suffix}.bin' if layer else f'xres{t}.bin')
            inputs.append(read_capture(inp,spec.hidden))
            c=read_attention(p,E,spec.num_heads,spec.num_kv_heads,spec.head_dim,'float32')
            c.update(read_boundaries(p,E,spec.hidden));c['xn']=read_candidate_norm(p,g);captures.append(c)
            rp=out/f'y_res{layer}{suffix}.bin';read_capture(rp,spec.hidden)
            for f in (p,rp):
                hashes[f.name]=digest(f)
                if mode=='bf16':control &= f.read_bytes()==(fixture/f.name).read_bytes()
        kv=out/f'kv_L{layer}.bin'
        if kv.stat().st_size!=meta['tokens']*m['layout']['kv_row']:raise ValueError('KV capture size mismatch')
        hashes[kv.name]=digest(kv)
        rr,wh=score_layer(spec,model,layer,inputs,captures);rows.extend(rr);weight_hashes.update(wh)
    result=dict(identity,passed=all(r['passed'] for r in rows) and control,control_identical=control if mode=='bf16' else None,
                scope='Isolated full ax with forced retained incoming residuals and causal per-layer KV; not full-model propagation',
                audit_sha256=digest(out/'audit.json'),capture_sha256=hashes,weight_sha256=weight_hashes,results=rows)
    (out/'score.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS' if result['passed'] else 'FAIL',mode,len(rows),'cases; control identical:',result['control_identical'])
    return 0 if result['passed'] else 1


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,RuntimeError) as e:print(f'FAIL: {e}',file=sys.stderr);sys.exit(1)
