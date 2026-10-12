"""Host-only layout, provenance and boundary checks for isolated attention replay."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'open_kernels'),str(ROOT/'open_kernels/model')]
from q4nx import bf16_to_f32


def geometry(spec,tokens):
    from recipes.qwen35 import recipe
    a=recipe(spec).attn
    if spec.family!='qwen35' or a is None or a.RB!=1 or not a.VEXP or a.NHL!=a.KVH or tokens<1:
        raise ValueError('requires split qwen35 attention, RB=1, VEXP=1, NHL=KVH and positive tokens')
    qb=a.QW*4//a.E_A;ob=a.NHL*a.HD*4//a.E_A;mb=(2*a.MLS*4+a.E_A-1)//a.E_A
    return dict(tokens=tokens,heads=a.NH,kv_heads=a.KVH,dim=a.HD,cores=a.ACORES,
                local_heads=a.NHL,mls=a.MLS,ea=a.E_A,qw=a.QW,kw=a.KVW,
                input_row=2*a.E_A+8*a.QW+8*a.KVW,qblocks=qb,oblocks=ob,
                debug_blocks=qb+ob+mb,debug_core=(qb+ob+mb)*a.E_A)


def pack_input(captures,meta,ptab,g):
    if len(captures)!=g['tokens'] or len(meta)!=g['ea'] or len(ptab)!=g['tokens']*g['ea']:
        raise ValueError('input coverage/metadata size mismatch')
    rows=[]
    for t,c in enumerate(captures):
        parts=[meta,ptab[t*g['ea']:(t+1)*g['ea']]]
        for name,n in [('q',g['heads']),('k',g['kv_heads']),('v',g['kv_heads']),('gate',g['heads'])]:
            a=np.asarray(c[name],np.float32)
            if a.shape!=(n,g['dim']) or not np.isfinite(a).all():raise ValueError('invalid projection shape/finite')
            parts.append(a.tobytes())
        row=b''.join(parts)
        if len(row)!=g['input_row']:raise ValueError('input row size mismatch')
        rows.append(row)
    return b''.join(rows)


def read_debug(path,g):
    data=path.read_bytes();t,c,b=g['tokens'],g['cores'],g['debug_core']
    if len(data)!=t*c*b:raise ValueError('debug size mismatch')
    raw=np.frombuffer(data,np.uint8).reshape(t,c,b)
    n=g['qblocks']*g['ea'];end=n+g['oblocks']*g['ea']
    qs=bf16_to_f32(raw[:,:,:n].copy().view(np.uint16)).reshape(t,c,2,g['qw'])
    qs=qs[:,:,0].astype(np.float64)+qs[:,:,1].astype(np.float64)
    o=raw[:,:,n:end].copy().view(np.float32).reshape(t,g['heads'],g['dim']).astype(np.float64)
    ml=raw[:,:,end:end+g['mls']*8].copy().view(np.float32).reshape(t,c,2,g['mls']).astype(np.float64)
    if not all(np.isfinite(x).all() for x in (qs,o,ml)):raise ValueError('non-finite debug')
    if not np.array_equal(qs,np.broadcast_to(qs[:,:1],qs.shape)):raise ValueError('Q copies differ across cores')
    return dict(q_scaled=qs[:,0].reshape(t,g['heads'],g['dim']),oacc=o,ml=ml)


def conditional_attention(q_scaled,kv):
    """FP64 causal softmax/PV on captured hardware Q (already scaled) and BF16 K/V."""
    t,nh,hd=q_scaled.shape;kh=kv.shape[2];out=np.empty_like(q_scaled)
    if kv.shape!=(t,2,kh,hd) or nh%kh or not np.isfinite(kv).all() or not np.isfinite(q_scaled).all():
        raise ValueError('invalid conditional attention inputs')
    for pos in range(t):
        for h in range(nh):
            k=h//(nh//kh);s=kv[:pos+1,0,k]@q_scaled[pos,h];p=np.exp(s-s.max());p/=p.sum()
            out[pos,h]=p@kv[:pos+1,1,k]
    return out


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify(out,expected_files):
    audit=json.loads((out/'audit.json').read_text())
    if set(audit['sha256'])!={str(p.resolve()) for p in expected_files}:raise ValueError('input hash coverage mismatch')
    for name,h in audit['sha256'].items():
        if digest(name)!=h:raise ValueError(f'input hash mismatch: {name}')
    return audit


def validate_build(meta,mode,tokens):
    if meta['mode']!=mode:raise ValueError('build mode mismatch')
    if meta['tokens']!=tokens:raise ValueError('build tokens mismatch')
