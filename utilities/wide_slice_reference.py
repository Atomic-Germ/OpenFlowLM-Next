"""Synthetic packing and independent FP64 layer oracles; never used during NPU runs."""
import numpy as np
from ml_dtypes import bfloat16
from recipes import pack, qwen35
from recipes.spec import LINEAR
from q4_1_pack import random_q4_1_blocks, pack_q4_1_pool, pool_reference
from wide_attention_reference import norm_rope, decode
from wide_deltanet_reference import ab_reference, glue_reference, step_reference

H,F,QW,KVW,NCH,VW = 5120,17408,6144,1024,10240,6144


def norm(x,w):
    return norm_rope(x,w,np.empty(0)).astype(np.float32).astype(bfloat16)


def silu(x):
    x = x.astype(np.float64)
    return x*np.exp(-np.logaddexp(0,-x))


def pack_linear(s,l,rng,pool):
    plan = qwen35.pack_plan(s)['layer_types'][LINEAR]
    const = np.zeros(l.C_BYTES,np.uint8)
    class Tensor:
        def __init__(self,raw): self.data=raw
        def raw(self,name): return self.data
    dims = dict(up=(F,H),gate=(F,H),down=(H,F),qkv=(NCH,H),z=(VW,H),out=(H,VW))
    suffixes = dict(up='mlp.up_proj.weight',gate='mlp.gate_proj.weight',down='mlp.down_proj.weight',
                    qkv='linear_attn.qkv_proj.weight',z='self_attn.gate_proj.weight',out='linear_attn.ssm_out_proj.weight')
    for name,(n,k) in dims.items():
        blocks = random_q4_1_blocks(n,k,rng,scale=.002)
        source = Tensor(pack_q4_1_pool(blocks,rs=1).tobytes())
        dest = const if name=='out' else pool
        op = next(op for op in plan['consts' if name=='out' else 'pool'] if op['tensor'].endswith(suffixes[name]))
        pack.apply_op(op,source,0,dest)
        expected = pack_q4_1_pool(blocks,rs=2)
        if not np.array_equal(dest[op['dst']:op['dst']+len(expected)],expected):
            raise ValueError(f'linear production packing mismatch: {name}')
    params = dict(lnw=rng.uniform(.8,1.2,H).astype(bfloat16),postw=rng.uniform(.8,1.2,H).astype(bfloat16),
                  nw=rng.uniform(.8,1.2,128).astype(bfloat16),weights=rng.normal(0,.015,(2,48,H)).astype(bfloat16),
                  a=-rng.uniform(.01,.08,48).astype(np.float32),dt=rng.uniform(-2.5,-1.5,48).astype(np.float32),
                  convw=rng.normal(0,.25,(4,NCH)).astype(bfloat16))
    names = {'input_layernorm.weight':params['lnw'],'post_attention_layernorm.weight':params['postw'],
             'ssm_norm.weight':params['nw'],'ssm_alpha_proj.bf16.weight':params['weights'][0],
             'ssm_beta_proj.bf16.weight':params['weights'][1],'ssm_a':params['a'],'ssm_dt.bias':params['dt'],
             'ssm_conv1d.weight':params['convw']}
    for op in plan['consts']:
        if op['op']=='std_perm': continue
        data = next(v for key,v in names.items() if op['tensor'].endswith(key))
        pack.apply_op(op,Tensor(data.tobytes()),0,const)
    return const,params


def projection(pool,const,l,kind,name,x):
    matrices = dict(up=(pool,l.POOL_FFN_UP,F,H),gate=(pool,l.POOL_FFN_GATE,F,H),down=(pool,l.POOL_FFN_DOWN,H,F))
    if kind==LINEAR:
        matrices.update(qkv=(pool,l.POOL_QKV,NCH,H),z=(pool,l.POOL_Z,VW,H),out=(const,l.C_WOUT,H,VW))
    else:
        matrices.update(q=(pool,l.POOL_Q,QW,H),k=(pool,l.POOL_K,KVW,H),v=(pool,l.POOL_V,KVW,H),
                        agate=(pool,l.POOL_GATE,QW,H),out=(pool,l.POOL_O,H,QW))
    storage,offset,n,k = matrices[name]
    return pool_reference(storage[offset:offset+n*k//8192*5120],x.astype(bfloat16),n,k,rs=2)


def evaluate(x,pool,const,l,kind,params,state,cs=None,pos=None):
    proj = lambda name,a: projection(pool,const,l,kind,name,a)
    xn = norm(x,params['lnw'])
    refs = dict(x=x,xn=xn)
    if kind==LINEAR:
        qkv,z = proj('qkv',xn),proj('z',xn)
        ab = ab_reference(xn,params['weights'],params['a'],params['dt']).astype(np.float32)
        conv,vec = glue_reference(qkv,state['conv'],params['convw'],ab)
        so,o = step_reference(state['so'],vec.astype(np.float32))
        so,o = so.astype(np.float32),o.astype(np.float32)
        a = o.astype(np.float64)
        og = (a/np.sqrt(np.mean(a*a,axis=1,keepdims=True)+1e-6)*params['nw'].astype(np.float64)*silu(z.reshape(48,128))).astype(np.float32).astype(bfloat16).ravel()
        state = dict(conv=conv,so=so)
        refs.update(qkv=qkv,z=z,ab=ab,conv=conv,vec=vec,so=so,o=o)
    else:
        qg = np.stack([proj('q',xn),proj('agate',xn)]).reshape(2,24,256)
        kvn = np.stack([proj('k',xn),proj('v',xn)]).reshape(2,4,256)
        new,og = decode(qg,kvn,params['norms'],cs,state,pos)
        og = og.astype(np.float32).astype(bfloat16).ravel()
        state[pos] = new
        refs.update(qg=qg,kvn=kvn,new=new)
    out = proj('out',og)
    res1 = (out+x).astype(np.float32)
    xm = norm(res1,params['postw'])
    up,gate = proj('up',xm),proj('gate',xm)
    h = (up.astype(np.float64)*silu(gate)).astype(np.float32)
    fo = proj('down',h)
    refs.update(og=og,projout=out,res1=res1,xm=xm,fo=fo,y=(fo+res1).astype(np.float32))
    return refs,state
