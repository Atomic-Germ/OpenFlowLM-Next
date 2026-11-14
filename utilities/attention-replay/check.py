"""Prepare or score low-memory production-attention replay with a FP32 finish experiment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from replay_common import geometry,pack_input,read_debug,conditional_attention,digest,verify,validate_build
from build import source_hash
from attention_heads import attention_reference,compare_heads,read_attention
from decode_bundle import load_bundle
from decode_reference import load_reference,read_capture,sfx
from q4nx import Q4NX,bf16_to_f32,f32_to_bf16
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.spec import FULL
from replica_qwen35 import rms,rope


def score_layer(spec,g,captures,qn,kn,output,kv,dbg):
    a={n:np.stack([r[n] for r in captures]) for n in captures[0]}
    ref=attention_reference(a['q'],a['k'],a['v'],a['gate'],qn,kn,spec.norm_eps,spec.rotary_dim,spec.rope_inv_freq())
    primary=compare_heads(output,ref)
    conditional=conditional_attention(dbg['q_scaled'],kv)
    e=np.exp(-np.abs(a['gate']));gate=np.where(a['gate']>=0,1/(1+e),e/(1+e))
    sums=dbg['ml'][:,:,1,:g['local_heads']].reshape(g['tokens'],g['heads'],1)
    if np.any(sums<=0):raise ValueError('invalid softmax sum')
    obs=dbg['oacc']/sums
    qr=rms(a['q'],spec.norm_eps)*qn;kr=rms(a['k'],spec.norm_eps)*kn
    for t in range(g['tokens']):
        qr[t]=rope(qr[t],t,spec.rotary_dim,spec.rope_inv_freq())
        kr[t]=rope(kr[t],t,spec.rotary_dim,spec.rope_inv_freq())
    return dict(passed=all(r['passed'] for r in primary),results=primary,
        production_output_byte_identical=output.astype(np.float32).tobytes()==a['og'].astype(np.float32).tobytes(),
        diagnostics=dict(q_scaled_maxabs=float(np.abs(dbg['q_scaled']-qr/np.sqrt(g['dim'])).max()),
            k_maxabs=float(np.abs(kv[:,0]-kr).max()),v_maxabs=float(np.abs(kv[:,1]-a['v']).max()),
            softmax_pv_from_captured_qkv=compare_heads(obs,conditional),
            output_from_captured_qkv=compare_heads(output,conditional*gate),
            output_from_captured_kv=compare_heads(output,conditional_attention(qr/np.sqrt(g['dim']),kv)*gate)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('model-dir','kernel-dir','fixture','out','build'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--mode',choices=['baseline','fp32'],required=True)
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
    if (build['geometry']!=g or build['production_build_key']!=m['build_key'] or
            build['source_hash']!=source_hash(spec,int(a.mode=='fp32'),meta['tokens'])):
        raise ValueError('build source/geometry mismatch')
    for n in ('final.xclbin','insts.bin'):
        if digest(b/n)!=build['sha256'][n]:raise ValueError('build binary hash mismatch')
    model=Q4NX(a.model_dir/'model.q4nx');ptab_path=a.fixture/'ptab.bin';ptab=ptab_path.read_bytes()
    if len(ptab)!=meta['max_ctx']*g['ea']:raise ValueError('position table size mismatch')
    it=g['tokens']*g['input_row'];kt=g['tokens']*g['kw']*4;ot=g['tokens']*g['qw']*4
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
        audit=dict(mode=a.mode,layers=layers,geometry=g,sha256={str(p.resolve()):digest(p) for p in files})
        (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(out/'run.cfg');return 0
    audit=verify(out,files)
    if audit['mode']!=a.mode or audit['layers']!=layers or audit['geometry']!=g or (out/'run.cfg').read_text()!=cfg:
        raise ValueError('requested replay differs from pinned configuration')
    records=[];hashes={}
    for l,cs,qn,kn,data in inputs:
        if (out/f'input_L{l}.bin').read_bytes()!=data:raise ValueError('input differs from fixture')
        output=read_capture(out/f'output_L{l}.bin',g['tokens']*g['qw']).reshape(g['tokens'],g['heads'],g['dim'])
        kvraw=(out/f'kv_L{l}.bin').read_bytes()
        if len(kvraw)!=kt:raise ValueError('KV size mismatch')
        kv=bf16_to_f32(np.frombuffer(kvraw,np.uint16)).astype(np.float64).reshape(g['tokens'],2,g['kv_heads'],g['dim'])
        dbg=read_debug(out/f'debug_L{l}.bin',g)
        r=score_layer(spec,g,cs,qn,kn,output,kv,dbg)
        if a.mode=='baseline':r['passed'] &= r['production_output_byte_identical']
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
