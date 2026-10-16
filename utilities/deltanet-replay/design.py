"""One production-layout DeltaNet head; two state passes, separate output BOs."""
from pathlib import Path
import hashlib
import numpy as np
import aie.iron as iron
from aie.iron import Buffer,CompileTime,In,Out,ObjectFifo,Program,Runtime,TaskGroup,Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from ironutil import include_dirs
HERE=Path(__file__).parent

@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def replay(s:In,v:In,so:Out,o:Out,*,source_hash:CompileTime[int]=0):
    elem=np.ndarray[(1280,),np.dtype[np.float32]]
    result=np.ndarray[(64,),np.dtype[np.float32]]
    state=np.ndarray[(130*128,),np.dtype[np.float32]]
    output=np.ndarray[(128,),np.dtype[np.float32]]
    wi=ObjectFifo(elem,name='w',depth=2);yo=ObjectFifo(result,name='y',depth=2)
    inc=include_dirs()+[str(HERE)]
    def fn(name,args):
        return ExternalFunction(name,source_file=str(HERE/(name+'.cc')),arg_types=args,
                                include_dirs=inc,compile_flags=['-Os','-DDNX_ROWS=10'])
    vc=fn('dnx_vcopy',[elem,elem]);p1=fn('dnx_pass1',[elem,elem,np.int32])
    de=fn('dnx_delta',[elem]);row=fn('dnx_row',[elem,elem,result,np.int32,np.int32])
    of=fn('dnx_ofin',[elem,result,np.int32])
    def body(w,y,ds,vc,p1,de,row,of):
        e=w.acquire(1);vc(e,ds);w.release(1)
        for b in range_(13):
            e=w.acquire(1);p1(e,ds,b);w.release(1)
        de(ds)
        for b in range_(13):
            e=w.acquire(1)
            for j in range_(20):
                z=y.acquire(1);row(e,ds,z,b,j);y.release(1)
            w.release(1)
        for h in range_(2):
            z=y.acquire(1);of(ds,z,h);y.release(1)
    worker=Worker(body,fn_args=[wi.cons(),yo.prod(),Buffer(elem,name='ds'),vc,p1,de,row,of],
                  tile=Tile(0,2),stack_size=6144)
    def tap(n):return TensorAccessPattern((1,n),0,[1,1,1,n],[0,0,0,1])
    def sequence(s,v,so,o,w,y):
        end=TaskGroup();inputs=TaskGroup()
        y.drain(so,tap=tap(130*128),wait=True,group=end)
        y.drain(o,tap=tap(128),wait=True,group=end)
        w.fill(v,tap=tap(1280),wait=True,group=inputs)
        w.fill(s,tap=tap(130*128),wait=True,group=inputs)
        w.fill(s,tap=tap(130*128),wait=True,group=inputs)
        inputs.finish();end.finish()
    rt=Runtime(sequence,[state,elem,state,output,wi.prod(),yo.cons()])
    return Program(iron.get_current_device(),rt,workers=[worker]).resolve_program()

DESIGN=replay
SPECIALIZE={'source_hash':int(hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.cc','.h'))).hexdigest()[:8],16)}
