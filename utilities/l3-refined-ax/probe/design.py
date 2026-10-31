"""Observe integrated entry tables and FP32 sum independently at one causal input."""
from pathlib import Path
import hashlib
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from ironutil import include_dirs
HERE=Path(__file__).parent
ROOT=next(p for p in HERE.parents if (p/"open_kernels").is_dir())

@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def probe(x: In, pairs: In, out: Out, table: Out, *, source_hash: CompileTime[int]):
    xt=np.ndarray[(2048,),np.dtype[bfloat16]]
    yt=np.ndarray[(64,),np.dtype[np.float32]]
    tt=np.ndarray[(23040,),np.dtype[np.uint8]]
    inputs=np.ndarray[(5120,),np.dtype[np.float32]]
    outputs=np.ndarray[(14336,),np.dtype[np.float32]]
    xi=ObjectFifo(xt,name='x',depth=1)
    pi=ObjectFifo(yt,name='pairs',depth=2)
    yo=ObjectFifo(yt,name='y',depth=1);to=ObjectFifo(tt,name='table',depth=1)
    def fn(name,args):
        return ExternalFunction(name,source_file=str(HERE/(name+'.cc')),arg_types=args,
                                include_dirs=include_dirs()+[str(HERE),str(ROOT/'open_kernels/designs/gemv_q4')],compile_flags=['-Os'])
    prep=fn('refine_prep',[xt,tt,np.int32]);summation=fn('refine_sum',[yt,yt]);copy=fn('copy64',[yt,yt])
    def body(x,pairs,y,table,prep,summation,copy):
        tab=table.acquire(1)
        for i in range_(5):
            e=x.acquire(1);prep(e,tab,i);x.release(1)
        table.release(1)
        for i in range_(224):
            pair=pairs.acquire(2);o=y.acquire(1)
            copy(pair[0],o);summation(o,pair[1])
            y.release(1);pairs.release(2)
    worker=Worker(body,fn_args=[xi.cons(),pi.cons(),yo.prod(),to.prod(),prep,summation,copy],tile=Tile(0,2),stack_size=6144)
    def sequence(x,pairs,out,table,xp,pp,yc,tc):
        tg=TaskGroup()
        xp.fill(x,group=tg);pp.fill(pairs,group=tg)
        yc.drain(out,wait=True,group=tg);tc.drain(table,wait=True,group=tg);tg.finish()
    rt=Runtime(sequence,[inputs,np.ndarray[(28672,),np.dtype[np.float32]],outputs,tt,xi.prod(),pi.prod(),yo.cons(),to.cons()])
    return Program(iron.get_current_device(),rt,workers=[worker]).resolve_program()

DESIGN=probe
SPECIALIZE=dict(source_hash=int(hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.cc','.h'))).hexdigest()[:8],16))
