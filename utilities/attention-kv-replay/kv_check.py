"""Prepare or score isolated paired-BF16 KV replay with FP32 output."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from kv_common import geometry,read_kv
from replay_common import pack_input,read_debug,digest,verify,validate_build
from kv_build import source_hash
# Reuse the independent oracle and boundary diagnostics unchanged.
import importlib.util
from kv_common import BASE
_score_spec=importlib.util.spec_from_file_location("attention_base_score",BASE/"check.py")
_score_module=importlib.util.module_from_spec(_score_spec);_score_spec.loader.exec_module(_score_module)
score_layer=_score_module.score_layer
from attention_heads import read_attention
from decode_bundle import load_bundle
from decode_reference import load_reference,read_capture,sfx
from q4nx import Q4NX,f32_to_bf16
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.spec import FULL
from recipes.cache import build_key


def validate_sources(build,manifest,fixture,spec,g,mode):
    # Retained projections belong to their original export. This isolated replay
    # has its own current source identity; it does not regenerate those projections.
    if any(manifest[k]!=fixture[k] for k in ('build_key','spec_hash')):
        raise ValueError('fixture identity mismatch')
    if (build['geometry']!=g or build['production_build_key']!=build_key(spec) or
            build['source_hash']!=source_hash(spec,mode,g['tokens'])):
        raise ValueError('build source/geometry mismatch')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('model-dir','kernel-dir','fixture','out','build'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--mode',choices=['bf16','pair'],required=True)
    ap.add_argument('--layers',help='default: all covered full-attention layers')
    ap.add_argument('--score',action='store_true');a=ap.parse_args()
    if a.out.resolve()==a.fixture.resolve():raise ValueError('separate replay output directory required')
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'score.json').unlink(missing_ok=True)
    spec=spec_from_model_dir(a.model_dir);meta=load_reference(a.fixture)
    m=load_bundle(a.kernel_dir,manifest(spec,max_ctx=meta['max_ctx']))
    if any(m[k]!=meta[k] for k in ('build_key','spec_hash')):raise ValueError('fixture identity mismatch')
    if meta['hidden']!=spec.hidden or meta['layers']>spec.num_layers:raise ValueError('fixture dimensions mismatch')
    layers=([l for l in range(meta['layers']) if spec.layer_types[l]==FULL] if a.layers is None
            else [int(n) for n in a.layers.split(',')])
    if not layers or len(layers)!=len(set(layers)) or any(l<0 or l>=meta['layers'] or spec.layer_types[l]!=FULL for l in layers):
        raise ValueError('invalid layer coverage')
    g=geometry(spec,meta['tokens']);L=layout(spec,max_ctx=meta['max_ctx']);b=a.build.resolve();out=a.out.resolve()
    build=json.loads((b/'build.json').read_text());validate_build(build,a.mode,meta['tokens'])
    validate_sources(build,m,meta,spec,g,a.mode)
    for n in ('final.xclbin','insts.bin'):
        if digest(b/n)!=build['sha256'][n]:raise ValueError('build binary hash mismatch')
    model=Q4NX(a.model_dir/'model.q4nx');ptab_path=a.fixture/'ptab.bin';ptab=ptab_path.read_bytes()
    if len(ptab)!=meta['max_ctx']*g['ea']:raise ValueError('position table size mismatch')
    it=g['tokens']*g['input_row'];kt=g['tokens']*g['kv_row_bytes'];ot=g['tokens']*g['qw']*4
    dt=g['tokens']*g['cores']*g['debug_core']
    cfg=['device',f'xclbin P {b}/final.xclbin',f'kernelx replay P {b}/insts.bin',
         f'buf input {it}',f'buf kv {kt}',f'buf output {ot}',f'buf debug {dt}']
    files=[b/'build.json',b/'final.xclbin',b/'insts.bin',a.fixture/'decode_reference.json',ptab_path,
           a.model_dir/'config.json',out/'run.cfg',out/'zero.bin']
    inputs=[]
    for l in layers:
        pre=f'model.layers.{l}.self_attn.';qn=model.bf16(pre+'q_norm.weight');kn=model.bf16(pre+'k_norm.weight')
        const_path=a.fixture/f'consts_{l}.bin';const=const_path.read_bytes()
        if len(const)!=L.CA_BYTES:raise ValueError('const size mismatch')
        norm=const[L.CA_META:L.CA_META+g['ea']]
        if norm[:spec.head_dim*4]!=f32_to_bf16(np.concatenate([qn,kn])).tobytes():raise ValueError('model norm differs from captured constants')
        paths=[a.fixture/f'y_act{l}{sfx(t)}.bin' for t in range(g['tokens'])]
        cs=[read_attention(p,L,g['heads'],g['kv_heads'],g['dim']) for p in paths]
        packed=pack_input(cs,norm,ptab[:g['tokens']*g['ea']],g)
        inp=out/f'input_L{l}.bin';files.extend([const_path,*paths,inp]);inputs.append((l,cs,qn,kn,packed))
        cfg += [f'load input {inp}',f'load kv {out}/zero.bin','run replay input kv output debug',
                f'dump kv {out}/kv_L{l}.bin {kt}',f'dump output {out}/output_L{l}.bin {ot}',
                f'dump debug {out}/debug_L{l}.bin {dt}']
    cfg='\n'.join(cfg)+'\n'
    if not a.score:
        for l,cs,qn,kn,data in inputs:(out/f'input_L{l}.bin').write_bytes(data)
        (out/'zero.bin').write_bytes(bytes(kt));(out/'run.cfg').write_text(cfg)
        audit=dict(mode=a.mode,layers=layers,geometry=g,fixture_build_key=m['build_key'],
                   replay_build_key=build['production_build_key'],sha256={str(p.resolve()):digest(p) for p in files})
        (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(out/'run.cfg');return 0
    audit=verify(out,files)
    if (audit['mode']!=a.mode or audit['layers']!=layers or audit['geometry']!=g or
            audit['fixture_build_key']!=m['build_key'] or audit['replay_build_key']!=build['production_build_key'] or
            (out/'run.cfg').read_text()!=cfg):
        raise ValueError('requested replay differs from pinned configuration')
    records=[];hashes={}
    for l,cs,qn,kn,data in inputs:
        if (out/f'input_L{l}.bin').read_bytes()!=data:raise ValueError('input differs from fixture')
        output=read_capture(out/f'output_L{l}.bin',g['tokens']*g['qw']).reshape(g['tokens'],g['heads'],g['dim'])
        kv=read_kv(out/f'kv_L{l}.bin',g,a.mode)
        dbg=read_debug(out/f'debug_L{l}.bin',g)
        r=score_layer(spec,g,cs,qn,kn,output,kv,dbg)
        records.append(dict(layer=l,**r))
        for name in ('output','kv','debug'):
            p=out/f'{name}_L{l}.bin';hashes[p.name]=digest(p)
    result=dict(passed=all(r['passed'] for r in records),mode=a.mode,scope='isolated attention on retained projections; no full-model propagation',
                audit_sha256=digest(out/'audit.json'),capture_sha256=hashes,results=records)
    (out/'score.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False));return 0 if result['passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,RuntimeError) as e:print(f'FAIL: {e}',file=sys.stderr);sys.exit(1)
