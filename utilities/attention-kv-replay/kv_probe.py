"""Isolated paired-BF16 KV experiment; both modes retain FP32 gated output.

KV_PAIR=0 stores zero low planes and uses production row arithmetic as control.
KV_PAIR=1 retains low planes through QK and PV. Four cache planes travel through
an EA-byte FIFO; the three-position hardware history has its own wider host ABI.
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

from kv_common import ROOT,BASE,geometry
sys.path.insert(0,str(ROOT/'open_kernels/designs/layer_x'))
from ironutil import Pipeline,include_dirs
from recipes.load import current_spec
from recipes.cache import build_key
import xlayer as X

S=current_spec();T=int(os.environ.get('REPLAY_TOKENS','3'));G=geometry(S,T);A=X.A
if A.KV_PAIR:
    raise ValueError("this standalone probe requires the earlier BF16 production ABI; "
                     "validate the integrated paired ABI with model/attention_heads.py")
MODE=int(os.environ.get('KV_PAIR','0'))
if MODE not in (0,1):raise ValueError('KV_PAIR must be 0 or 1')
HERE=Path(__file__).resolve().parent

def ty(n,dtype):return np.ndarray[(n,),np.dtype[dtype]]
def bt(total,off,n):return TensorAccessPattern((1,total),off,[1,1,1,n],[0,0,0,1])

@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def probe(inputs:In,kv:InOut,outputs:InOut,debug:InOut,*,srchash:CompileTime[int]):
    a=X.attn_types();inc=include_dirs()+[str(X.ATTN)]
    fns=X.attn_kernels(inc,a)
    wide=ty(2*A.KVW,X.bfloat16)
    def kf(sym,args):
        return ExternalFunction(sym,source_file=str(HERE/f'{sym}.cc'),arg_types=args,
            include_dirs=inc+[str(HERE)],compile_flags=X.ATTN_FLAGS+[f'-DKV_PAIR={MODE}'])
    fns[2]=kf('kv_k',[a['u8_1k'],a['b256'],a['f64'],a['f256'],wide,np.int32])
    fns[3]=kf('kv_v',[a['u8_1k'],wide,np.int32])
    fns[5]=kf('kv_step',[a['u8_1k']]*4+[a['fq'],a['foacc'],a['fml'],a['pb'],np.int32])
    fns[6]=kf('kv_step_new',[wide,wide,a['fq'],a['foacc'],a['fml'],np.int32])
    ogty=ty(A.KVW,np.float32)
    fns[-1]=ExternalFunction('probe_fin',source_file=str(BASE/'probe_fin.cc'),
        arg_types=[a['foacc'],a['fml'],a['u8_1k'],a['u8_1k'],ogty,np.int32],
        include_dirs=inc,compile_flags=X.ATTN_FLAGS+['-DREPLAY_FP32=1'])
    fd=ExternalFunction('probe_debug',source_file=str(BASE/'probe_debug.cc'),
        arg_types=[a['fq'],a['foacc'],a['fml'],a['b512'],np.int32],include_dirs=inc,compile_flags=X.ATTN_FLAGS)
    inp=ObjectFifo(a['u8_1k'],name='in',depth=4)
    ko=ObjectFifo(a['b512'],name='kvout',depth=2)
    # One output per token; depth 1 saves 4 KiB/core for the expanded KV scratch.
    og=[ObjectFifo(ogty,name=f'og{c}',depth=1) for c in range(A.ACORES)]
    dbg=[ObjectFifo(a['b512'],name=f'dbg{c}',depth=2) for c in range(1,A.ACORES)]
    def body(c):
        def run(ai,ki,oi,di,qn,kn,cs,qs,tmp,kout,vout,oacc,ml,pb,fm,fq,fk,fv,fi,fs,fsn,ff,df):
            e=ai.acquire(2);fm(e[0],e[1],qn,kn,cs,pb);ai.release(2)
            for h in range_(A.Q_AIN_ELEMS):
                e=ai.acquire(1);fq(e,qn,cs,qs,h);ai.release(1)
            for h in range_(A.K_AIN_ELEMS):
                e=ai.acquire(1);fk(e,kn,cs,tmp,kout,h);ai.release(1)
            for h in range_(A.K_AIN_ELEMS):
                e=ai.acquire(1);fv(e,vout,h);ai.release(1)
            if c==0:
                for plane in range_(2):
                    e=ki.acquire(1)
                    for j in range_(A.KVW):e[j]=kout[plane*A.KVW+j]
                    ki.release(1)
                for plane in range_(2):
                    e=ki.acquire(1)
                    for j in range_(A.KVW):e[j]=vout[plane*A.KVW+j]
                    ki.release(1)
            fi(oacc,ml)
            for row in range_(pb[1]):
                e=ai.acquire(4);fs(e[0],e[1],e[2],e[3],qs,oacc,ml,pb,c*A.NHL);ai.release(4)
            fsn(kout,vout,qs,oacc,ml,c*A.NHL)
            for _ in range(c*X.N_OG):ai.acquire(2);ai.release(2)
            for hp in range_(X.N_OG):
                e=ai.acquire(2);o=oi.acquire(1);ff(oacc,ml,e[0],e[1],o,hp);oi.release(1);ai.release(2)
            for _ in range((A.ACORES-1-c)*X.N_OG):ai.acquire(2);ai.release(2)
            for i in range_(G['debug_blocks']):
                e=di.acquire(1);df(qs,oacc,ml,e,i);di.release(1)
        return run
    workers=[]
    for c in range(A.ACORES):
        bufs=[Buffer(wide if n in ('kout','vout') else a[k],name=f'{n}{c}') for n,k in [('qn','b256'),('kn','b256'),('cs','f64'),
            ('qs','fq'),('tmp','f256'),('kout','b512'),('vout','b512'),('oacc','foacc'),('ml','fml'),('pb','pb')]]
        debug_prod=ko.prod() if c==0 else dbg[c-1].prod()
        workers.append(Worker(body(c),fn_args=[inp.cons(),debug_prod if c==0 else None,og[c].prod(),debug_prod]+bufs+fns+[fd],
                              tile=Tile(2+c,3),stack_size=X.STACK))
    it=ty(T*G['input_row'],np.uint8);kt=ty(T*4*A.KVW,np.uint16)
    ot=ty(T*A.QW,np.float32);dt=ty(T*A.ACORES*G['debug_core'],np.uint8)
    def seq(i,k,o,d,ip,kc,ocs,dcs):
        p=Pipeline(3)
        for t in range(T):
            p.drain(kc,k,bt(T*4*A.KVW,t*4*A.KVW,4*A.KVW))
            for c in range(A.ACORES):
                p.drain(ocs[c],o,bt(T*A.QW,t*A.QW+c*A.NHL*A.HD,A.NHL*A.HD))
                p.drain(kc if c==0 else dcs[c-1],d,bt(T*A.ACORES*G['debug_core'],(t*A.ACORES+c)*G['debug_core'],G['debug_core']))
            base=t*G['input_row']
            p.fill(ip,i,bt(T*G['input_row'],base,2*A.E_A+4*A.QW+8*A.KVW))
            p.fill(ip,k,bt(T*4*A.KVW,0,max(t,1)*4*A.KVW))
            p.fill(ip,i,bt(T*G['input_row'],base+2*A.E_A+4*A.QW+8*A.KVW,4*A.QW))
            p.finish()
    rt=Runtime(seq,[it,kt,ot,dt,inp.prod(tile=Tile(0,0)),ko.cons(tile=Tile(1,0)),
                   [f.cons(tile=Tile(2+c,0)) for c,f in enumerate(og)],
                   [f.cons(tile=Tile(3+c,0)) for c,f in enumerate(dbg)]])
    return Program(iron.get_current_device(),rt,workers=workers).resolve_program()

from kv_build import source_hash
DESIGN=probe
SPECIALIZE=dict(srchash=int(source_hash(S,'pair' if MODE else 'bf16',T)[:8],16))
