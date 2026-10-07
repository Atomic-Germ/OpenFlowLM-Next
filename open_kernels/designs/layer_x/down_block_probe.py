"""One real K4096/32-row segment; production Q4 arithmetic with block snapshots."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, TaskGroup
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE=Path(__file__).parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT))
from ironutil import include_dirs


@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def probe(weights: In, activation: In, output: Out, *, source_hash: CompileTime[int]=0):
    wt=np.ndarray[(5120,),np.dtype[np.uint8]]
    xt=np.ndarray[(1024,),np.dtype[np.float32]]
    yt=np.ndarray[(1120,),np.dtype[np.float32]]
    tt=np.ndarray[(4*4096+4096//4+4096//16+512,),np.dtype[np.uint8]]
    mt=np.ndarray[(32,),np.dtype[np.float32]]
    inc=include_dirs()+[str(HERE.parent/'gemv_q4')]
    flags=['-Os','-DGEMV_Q4_CORRECTION=1','-DGEMV_Q4_PRODUCT_CORRECTION=1',
           '-DGEMV_Q4_BLOCK_CARRY=1','-DGEMV_Q4_SEGMENT_CARRY=1','-DGEMV_Q4_BLOCK_TRACE=1']
    prep=ExternalFunction('down_block_prep',source_file=str(HERE/'down_block_prep.cpp'),
                          arg_types=[xt,tt,np.int32],include_dirs=inc,compile_flags=flags)
    step=ExternalFunction('down_block_trace',source_file=str(HERE/'down_block_trace.cpp'),
                          arg_types=[wt,tt,mt,yt,np.int32],include_dirs=inc,compile_flags=flags)
    w,x,y=ObjectFifo(wt,name='w',depth=2),ObjectFifo(xt,name='x',depth=2),ObjectFifo(yt,name='y',depth=2)
    tab,ms=Buffer(tt,name='tab'),Buffer(mt,name='ms')

    def core(wi,xi,yo,tab,ms,prep,step):
        for i in range_(4):
            xe=xi.acquire(1);prep(xe,tab,i);xi.release(1)
        for kt in range_(16):
            we=wi.acquire(1);ye=yo.acquire(1)
            step(we,tab,ms,ye,kt)
            wi.release(1);yo.release(1)

    worker=Worker(core,fn_args=[w.cons(),x.cons(),y.prod(),tab,ms,prep,step],tile=Tile(0,2),stack_size=0x1800)
    def tap(n): return TensorAccessPattern((1,n),0,[1,1,1,n],[0,0,0,1])
    def sequence(a_w,a_x,a_y,wp,xp,yc):
        group=TaskGroup()
        yc.drain(a_y,tap=tap(16*1120),wait=True,group=group)
        xp.fill(a_x,tap=tap(4096),wait=True,group=group)
        wp.fill(a_w,tap=tap(16*5120),wait=True,group=group)
        group.finish()
    rt=Runtime(sequence,[np.ndarray[(16*5120,),np.dtype[np.uint8]],
                         np.ndarray[(4096,),np.dtype[np.float32]],
                         np.ndarray[(16*1120,),np.dtype[np.float32]],
                         w.prod(tile=Tile(0,0)),x.prod(tile=Tile(1,0)),y.cons(tile=Tile(0,0))])
    return Program(iron.get_current_device(),rt,workers=[worker]).resolve_program()


DESIGN=probe
paths=[Path(__file__),HERE/'down_block_prep.cpp',HERE/'down_block_trace.cpp',ROOT/'ironutil.py',
       *sorted((HERE.parent/'gemv_q4').glob('*.h')),*sorted((ROOT/'include').glob('*.h'))]
SPECIALIZE={'source_hash':int(hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()[:8],16)}
