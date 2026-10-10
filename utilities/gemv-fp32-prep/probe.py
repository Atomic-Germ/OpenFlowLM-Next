"""Single-core FP32 table preparation probe; no model or production export required."""
import hashlib
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction
from aie.utils import config

HERE=Path(__file__).resolve().parent
GEMV=HERE.parents[1]/'open_kernels/designs/gemv_q4'

@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def probe(x: In, table: Out, *, srchash: CompileTime[int]):
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    inc=_include_dirs();root=Path(config.cxx_header_path())/'aie_kernels'
    inc += [str(root),str(root/_detect_arch()),str(GEMV)]
    xt=np.ndarray[(1024,),np.dtype[np.float32]]
    tt=np.ndarray[(2304,),np.dtype[np.uint8]]
    fn=ExternalFunction('probe',source_file=str(HERE/'probe.cc'),arg_types=[xt,tt],include_dirs=inc)
    rx=ObjectFifo(xt,name='x',depth=1);tx=ObjectFifo(tt,name='table',depth=1)
    def body(r,t,f):
        a=r.acquire(1);b=t.acquire(1);f(a,b);r.release(1);t.release(1)
    worker=Worker(body,fn_args=[rx.cons(),tx.prod(),fn],stack_size=0x1000)
    def seq(a,b,r,t):
        r.fill(a);t.drain(b,wait=True)
    return Program(iron.get_current_device(),Runtime(seq,[xt,tt,rx.prod(),tx.cons()]),workers=[worker]).resolve_program()

DESIGN=probe
SPECIALIZE=dict(srchash=int(hashlib.sha256((GEMV/'gemv_tab.h').read_bytes()+(HERE/'probe.cc').read_bytes()).hexdigest()[:8],16))
