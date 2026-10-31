"""240 causal positions, unmodified production paired-attention worker and kernels."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime,In,InOut,Program,Runtime
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'open_kernels'),str(ROOT/'open_kernels/designs/layer_x'),str(Path(__file__).parent)]
from ironutil import Pipeline,include_dirs
from recipes.load import current_spec
import xlayer as X
from build import geometry,sources
S=current_spec();G=geometry(S);T=G['tokens'];A=X.A


def ty(n):return np.ndarray[(n,),np.dtype[np.uint8]]
def bt(total,off,n):return TensorAccessPattern((1,total),off,[1,1,1,n],[0,0,0,1])


@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def replay(inputs:In,kv:InOut,output:InOut,*,source_hash:CompileTime[int]):
    a=X.attn_types();f=X.attn_kernels(include_dirs()+[str(X.ATTN)],a)
    ai,ao,og=X.attn_fifos(a)
    workers=X.attn_workers(ai,ao,og,a,f)
    ni,nk,no=[T*G[k] for k in ('input_row','kv_row','output_row')]
    def seq(i,k,o,ip,kc,ocs):
        p=Pipeline(3)
        for t in range(T):
            p.drain(kc,k,bt(nk,t*G['kv_row'],G['kv_row']))
            for c in range(A.ACORES):
                p.drain(kc if c==0 else ocs[c-1],o,bt(no,t*G['output_row']+c*A.NHL*A.HD*4,A.NHL*A.HD*4))
            start=t*G['input_row'];prefix=2*A.E_A+4*A.QW+8*A.KVW
            p.fill(ip,i,bt(ni,start,prefix))
            p.fill(ip,k,bt(nk,0,max(t,1)*G['kv_row']))
            p.fill(ip,i,bt(ni,start+prefix,4*A.QW))
            p.finish()
    rt=Runtime(seq,[ty(ni),ty(nk),ty(no),ai.prod(tile=Tile(0,0)),ao.cons(tile=Tile(1,0)),
                   [f.cons(tile=Tile(2+c,0)) for c,f in enumerate(og)]])
    return Program(iron.get_current_device(),rt,workers=workers).resolve_program()


DESIGN=replay
SPECIALIZE=dict(source_hash=int(hashlib.sha256(json.dumps(sources(S),sort_keys=True).encode()).hexdigest()[:8],16))
