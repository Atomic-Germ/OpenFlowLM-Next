"""Replay production split attention, retaining KV and softmax boundaries.

REPLAY_FP32=0 promotes the original BF16 finish, 1 stores its pre-round FP32.
No production kernel or ABI is changed. All positions run with hardware KV history.
"""
import os
from pathlib import Path
import sys
import numpy as np
import aie.iron as iron
from aie.iron import Buffer,CompileTime,In,InOut,ObjectFifo,Program,Runtime,Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

from replay_common import ROOT,geometry
sys.path.insert(0,str(ROOT/'open_kernels/designs/layer_x'))
from ironutil import Pipeline,include_dirs
from recipes.load import current_spec
from recipes.cache import build_key
import xlayer as X

S=current_spec();T=int(os.environ.get('REPLAY_TOKENS','3'));G=geometry(S,T);A=X.A
if A.KV_PAIR:
    raise ValueError("this standalone probe requires the earlier BF16 production ABI; "
                     "validate the integrated paired ABI with model/attention_heads.py")
MODE=int(os.environ.get('REPLAY_FP32','0'))
if MODE not in (0,1):raise ValueError('REPLAY_FP32 must be 0 or 1')
HERE=Path(__file__).resolve().parent

def ty(n,dtype):return np.ndarray[(n,),np.dtype[dtype]]
def bt(total,off,n):return TensorAccessPattern((1,total),off,[1,1,1,n],[0,0,0,1])

@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def probe(inputs:In,kv:InOut,outputs:InOut,debug:InOut,*,srchash:CompileTime[int]):
    a=X.attn_types();inc=include_dirs()+[str(X.ATTN)]
    fns=X.attn_kernels(inc,a)
    ogty=ty(A.KVW,np.float32)
    fns[-1]=ExternalFunction('probe_fin',source_file=str(HERE/'probe_fin.cc'),
        arg_types=[a['foacc'],a['fml'],a['u8_1k'],a['u8_1k'],ogty,np.int32],
        include_dirs=inc,compile_flags=X.ATTN_FLAGS+[f'-DREPLAY_FP32={MODE}'])
    fd=ExternalFunction('probe_debug',source_file=str(HERE/'probe_debug.cc'),
        arg_types=[a['fq'],a['foacc'],a['fml'],a['b512'],np.int32],include_dirs=inc,compile_flags=X.ATTN_FLAGS)
    inp=ObjectFifo(a['u8_1k'],name='in',depth=4)
    ko=ObjectFifo(a['b512'],name='kvout',depth=2)
    og=[ObjectFifo(ogty,name=f'og{c}',depth=2) for c in range(A.ACORES)]
    dbg=[ObjectFifo(a['b512'],name=f'dbg{c}',depth=2) for c in range(1,A.ACORES)]
    def body(c):
        def run(ai,ki,oi,di,qn,kn,cs,qs,tmp,kout,vout,oacc,ml,pb,fm,fq,fk,fv,fi,fs,fsn,ff,df):
            X._attn(ai,ki if c==0 else None,oi,qn,kn,cs,qs,tmp,kout,vout,oacc,ml,pb,
                    fm,fq,fk,fv,fi,fs,fsn,ff,None,None,c)
            for i in range_(G['debug_blocks']):
                e=di.acquire(1);df(qs,oacc,ml,e,i);di.release(1)
        return run
    workers=[]
    for c in range(A.ACORES):
        bufs=[Buffer(a[k],name=f'{n}{c}') for n,k in [('qn','b256'),('kn','b256'),('cs','f64'),
            ('qs','fq'),('tmp','f256'),('kout','b512'),('vout','b512'),('oacc','foacc'),('ml','fml'),('pb','pb')]]
        debug_prod=ko.prod() if c==0 else dbg[c-1].prod()
        workers.append(Worker(body(c),fn_args=[inp.cons(),debug_prod if c==0 else None,og[c].prod(),debug_prod]+bufs+fns+[fd],
                              tile=Tile(2+c,3),stack_size=X.STACK))
    it=ty(T*G['input_row'],np.uint8);kt=ty(T*2*A.KVW,np.uint16)
    ot=ty(T*A.QW,np.float32);dt=ty(T*A.ACORES*G['debug_core'],np.uint8)
    def seq(i,k,o,d,ip,kc,ocs,dcs):
        p=Pipeline(3)
        for t in range(T):
            p.drain(kc,k,bt(T*2*A.KVW,t*2*A.KVW,2*A.KVW))
            for c in range(A.ACORES):
                p.drain(ocs[c],o,bt(T*A.QW,t*A.QW+c*A.NHL*A.HD,A.NHL*A.HD))
                p.drain(kc if c==0 else dcs[c-1],d,bt(T*A.ACORES*G['debug_core'],(t*A.ACORES+c)*G['debug_core'],G['debug_core']))
            base=t*G['input_row']
            p.fill(ip,i,bt(T*G['input_row'],base,2*A.E_A+4*A.QW+8*A.KVW))
            p.fill(ip,k,bt(T*2*A.KVW,0,max(t,1)*2*A.KVW))
            p.fill(ip,i,bt(T*G['input_row'],base+2*A.E_A+4*A.QW+8*A.KVW,4*A.QW))
            p.finish()
    rt=Runtime(seq,[it,kt,ot,dt,inp.prod(tile=Tile(0,0)),ko.cons(tile=Tile(1,0)),
                   [f.cons(tile=Tile(2+c,0)) for c,f in enumerate(og)],
                   [f.cons(tile=Tile(3+c,0)) for c,f in enumerate(dbg)]])
    return Program(iron.get_current_device(),rt,workers=workers).resolve_program()

from build import source_hash
DESIGN=probe
SPECIALIZE=dict(srchash=int(source_hash(S,MODE,T)[:8],16))
