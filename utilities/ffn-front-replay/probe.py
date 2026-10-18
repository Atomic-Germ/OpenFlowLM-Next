"""Isolated FP32 RMSNorm -> up/gate/SiLU probe, with a BF16-prep control build.

No change to the production layer layout. Captures each projection independently.
"""
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'open_kernels'
sys.path.insert(0,str(ROOT))
from ironutil import Pipeline,include_dirs
from recipes.load import current_spec
from recipes.qwen35 import recipe
from recipes.qwen36moe import band_bytes
sys.path.insert(0,str(HERE))
from front_build import prep_flags,validate_spec,sources as build_sources
S=current_spec();R=recipe(S);H=S.hidden;F=S.intermediate;C=R.common;L=R.layout
if S.family!='qwen35' or 'ffn' in S.q8_roles or H%1024:
    raise ValueError('probe requires qwen35 Q4 FFN with hidden divisible by 1024')
MODE=int(os.environ.get('FRONT_FP32','1'))
validate_spec(S,MODE)
GEMV=ROOT/'designs/gemv_q4';LN=ROOT/'designs/ln'

def ty(n,dtype):return np.ndarray[(n,),np.dtype[dtype]]
def bt(total,off,n):return TensorAccessPattern((1,total),off,[1,1,1,n],[0,0,0,1])

@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def probe(pool:In, resnorm:In, norm:InOut, captures:InOut, *, srchash:CompileTime[int]):
    inc=include_dirs()+[str(GEMV),str(LN)]
    u=ty(H,bfloat16);x=ty(1024,np.float32);w=ty(C.CALL_BYTES,np.uint8)
    y=ty(64,np.float32);tab=ty(H*9//4,np.uint8);ms=ty(128,np.float32)
    flags=['-Os',f'-DGEMV_PER_CALL={C.PER_CALL}']+prep_flags(MODE)
    def ext(name,args):return ExternalFunction(name,source_file=str(HERE/(name+'.cc')),arg_types=args,include_dirs=inc,compile_flags=flags)
    prep=ext('front_prep',[x,tab,np.int32,np.int32])
    gms=ext('front_gms',[w,tab,ms,np.int32,np.int32,np.int32])
    emit=ext('front_emit',[ms,y,np.int32])
    nr=ExternalFunction('ln_nr32',source_file=str(LN/'ln_nr32.cc'),arg_types=[u,u,u,u,np.int32],include_dirs=inc,compile_flags=[f'-DLN_N={H}',f'-DLN_EPS={S.norm_eps:g}f'])
    ni=ObjectFifo(u,name='ni',depth=3);no=ObjectFifo(u,name='no',depth=1)
    wi=ObjectFifo(w,name='wi',depth=2);xi=ObjectFifo(x,name='xi',depth=2);yo=ObjectFifo(y,name='yo',depth=2)
    def norm_body(inp,out,fn):
        a=inp.acquire(3)
        for i in range(2):
            b=out.acquire(1);fn(a[0],a[1],a[2],b,i);out.release(1)
        inp.release(3)
    def gemv_body(weights,inp,out,t,s,p,g,e):
        for i in range_(H//1024):
            a=inp.acquire(1);p(a,t,H,i);inp.release(1)
        for _ in range_(F//64):
            for dst in (0,64):
                for group in range_(band_bytes(H)//C.CALL_BYTES):
                    a=weights.acquire(1);g(a,t,s,group,band_bytes(H)//5120,dst);weights.release(1)
            for part in range(3):
                b=out.acquire(1);e(s,b,part);out.release(1)
    workers=[Worker(norm_body,fn_args=[ni.cons(),no.prod(),nr],tile=Tile(0,3),stack_size=0x800),
             Worker(gemv_body,fn_args=[wi.cons(),xi.cons(),yo.prod(),Buffer(tab,name='tab'),Buffer(ms,name='ms'),prep,gms,emit],tile=Tile(0,2),stack_size=0x800)]
    pool_t=ty(L.POOL_BYTES,np.uint8);rn_t=ty(H*6,np.uint8);n_t=ty(H,np.float32);out_t=ty(F*3,np.float32)
    def seq(a,b,n,o,ip,oc,wp,xp,yc):
        p=Pipeline(3)
        p.drain(oc,n,bt(H,0,H));p.fill(ip,b,bt(H*6,0,H*6));p.finish()
        p.fill(xp,n,bt(H,0,H));p.drain(yc,o,bt(F*3,0,F*3))
        for band in range(F//64):
            for off in (L.POOL_FFN_UP,L.POOL_FFN_GATE):
                p.fill(wp,a,bt(L.POOL_BYTES,off+band*band_bytes(H),band_bytes(H)))
        p.finish()
    return Program(iron.get_current_device(),Runtime(seq,[pool_t,rn_t,n_t,out_t,ni.prod(),no.cons(),wi.prod(),xi.prod(),yo.cons()]),workers=workers).resolve_program()

sources=build_sources(S)
DESIGN=probe
SPECIALIZE=dict(srchash=int(hashlib.sha256(b''.join(p.read_bytes() for p in sources)+S.to_json().encode()+str(MODE).encode()).hexdigest()[:8],16))
