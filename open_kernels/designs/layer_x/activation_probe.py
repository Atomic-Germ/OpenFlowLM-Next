"""Diagnostic-only 32-lane replay of the production compensated SiLU helpers."""
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ironutil import Pipeline, include_dirs
SERIES=os.environ.get('PROBE_ACTIVATION_SERIES')=='1'


@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def probe(x: In,y: Out,*,key: CompileTime[int]):
    it=np.ndarray[(64,),np.dtype[np.float32]]
    ot=np.ndarray[(320,),np.dtype[np.float32]]
    fn=ExternalFunction('activation_probe',source_file=str(HERE/'activation_probe.cpp'),
                        arg_types=[it,ot],include_dirs=include_dirs()+[str(ROOT/'designs/gemv_q4')],
                        compile_flags=['-Os','-DGEMV_Q4_CORRECTION=1','-DGEMV_Q4_PRODUCT_CORRECTION=1',
                                       f'-DDENSE_ACT_SERIES={int(SERIES)}'])
    inp=ObjectFifo(it,name='input',depth=1);out=ObjectFifo(ot,name='output',depth=1)
    def body(i,o,f):
        a=i.acquire(1);b=o.acquire(1);f(a,b);o.release(1);i.release(1)
    worker=Worker(body,fn_args=[inp.cons(),out.prod(),fn],tile=Tile(0,2),stack_size=0x1800)
    def sequence(a,b,i,o):
        p=Pipeline(3)
        p.drain(o,b,TensorAccessPattern((1,320),0,[1,1,1,320],[0,0,0,1]))
        p.fill(i,a,TensorAccessPattern((1,64),0,[1,1,1,64],[0,0,0,1]));p.finish()
    rt=Runtime(sequence,[it,ot,inp.prod(),out.cons()])
    return Program(iron.get_current_device(),rt,workers=[worker]).resolve_program()


DESIGN=probe
files=[HERE/'activation_probe.cpp',HERE/'activation_probe.py',HERE/'dense_activation_carry.h',HERE/'sigmoid_series.h',
       ROOT/'include/vecmath.h',ROOT/'include/vecmath_precise.h',ROOT/'include/fp32_add_rne.h',
       ROOT/'designs/gemv_q4/gemv_tab.h']
SPECIALIZE={'key':int(hashlib.sha256(b''.join(p.read_bytes() for p in files)+str(SERIES).encode()).hexdigest()[:8],16)}
