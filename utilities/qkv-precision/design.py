"""One-core 10240x5120 Q4 projection with captured FP32 entry and table dump."""
from pathlib import Path
import hashlib
import numpy as np
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, Out, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from ironutil import include_dirs

HERE = Path(__file__).parent


@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def replay(w: In, x: In, y: Out, table: Out, *, source_hash: CompileTime[int]):
    wt = np.ndarray[(5120,), np.dtype[np.uint8]]
    xt = np.ndarray[(1024,), np.dtype[np.float32]]
    yt = np.ndarray[(64,), np.dtype[np.float32]]
    tt = np.ndarray[(11520,), np.dtype[np.uint8]]
    weights = np.ndarray[(32768000,), np.dtype[np.uint8]]
    inputs = np.ndarray[(5120,), np.dtype[np.float32]]
    outputs = np.ndarray[(10240,), np.dtype[np.float32]]
    wi = ObjectFifo(wt, name='w', depth=2)
    xi = ObjectFifo(xt, name='x', depth=1)
    yo = ObjectFifo(yt, name='y', depth=2)
    to = ObjectFifo(tt, name='table', depth=1)

    def fn(name, args):
        return ExternalFunction(name, source_file=str(HERE/(name+'.cc')), arg_types=args,
                                include_dirs=include_dirs()+[str(HERE)], compile_flags=['-Os'])

    prep = fn('prep', [xt, tt, np.int32])
    gemv = fn('gemv', [wt, tt, yt, np.int32])

    def body(w, x, y, table, prep, gemv):
        t = table.acquire(1)
        for block in range_(5):
            e = x.acquire(1); prep(e, t, block); x.release(1)
        for band in range_(160):
            result = y.acquire(1)
            for group in range_(40):
                e = w.acquire(1); gemv(e, t, result, group); w.release(1)
            y.release(1)
        table.release(1)

    worker = Worker(body, fn_args=[wi.cons(), xi.cons(), yo.prod(), to.prod(), prep, gemv],
                    tile=Tile(0, 2), stack_size=6144)

    def sequence(w, x, y, table, wp, xp, yc, tc):
        tg = TaskGroup()
        xp.fill(x, group=tg); wp.fill(w, group=tg)
        yc.drain(y, wait=True, group=tg); tc.drain(table, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [weights, inputs, outputs, tt, wi.prod(), xi.prod(), yo.cons(), to.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = replay
SPECIALIZE = dict(source_hash=int(hashlib.sha256(b''.join(
    p.read_bytes() for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.cc', '.h'))).hexdigest()[:8], 16))
