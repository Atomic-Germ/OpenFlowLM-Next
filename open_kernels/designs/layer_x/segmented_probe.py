"""Actual dense segmented down worker/schedule, with optional partial snapshots.

Full FFN mode runs the production up/gate/act/down body and schedule. Both modes
keep actual main-core scratch and pool/activation offsets, without other layers.
"""
import hashlib
from dataclasses import replace
import os
from pathlib import Path
import sys

import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, InOut, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs
import xcommon as X
from recipes.segmented_dense import segments
from recipes.wide_deltanet_layer import projection_table_bytes

FULL = os.environ.get('PROBE_FULL_FFN') == '1'
TRACE = FULL and os.environ.get('PROBE_FFN_TRACE') == '1'
DIAGNOSTIC = not FULL and os.environ.get('PROBE_PARTIALS', '1') == '1'
CORRECTION = os.environ.get('PROBE_FFN_CORRECTION') == '1'
PRODUCT_CORRECTION = os.environ.get('PROBE_PRODUCT_CORRECTION') == '1'
BLOCK_CARRY = os.environ.get('PROBE_BLOCK_CARRY') == '1'
SEGMENT_CARRY = os.environ.get('PROBE_SEGMENT_CARRY') == '1'
DOWN_RNE = os.environ.get('PROBE_DOWN_RNE') == '1'
if DOWN_RNE:
    if not SEGMENT_CARRY:
        raise ValueError('down RNE requires segment carry')
    X.OS.append('-DDENSE_DOWN_RNE=1')
ACTIVATION_CARRY = os.environ.get('PROBE_ACTIVATION_CARRY') == '1'
ACTIVATION_SERIES = os.environ.get('PROBE_ACTIVATION_SERIES') == '1'
if ACTIVATION_SERIES:
    if not ACTIVATION_CARRY:
        raise ValueError('activation series requires activation carry')
    X.OS.append('-DDENSE_ACT_SERIES=1')
    X.DYNAMIC_DOWN = True
    X.COMPACT_UP = True
if ACTIVATION_CARRY:
    if not FULL or not PRODUCT_CORRECTION:
        raise ValueError('activation carry requires full FFN product correction')
    X.OS.append('-DDENSE_ACT_CARRY=1')
    X.COMPACT_DOWN = True
if SEGMENT_CARRY:
    if not FULL or not BLOCK_CARRY or X.DS_FLOATS < 2 * X.FFN.DOWN_PC * 64:
        raise ValueError('segment carry requires full FFN block carry and two down scratch planes')
    X.SEGMENT_CARRY = True
    X.OS += ['-DGEMV_Q4_SEGMENT_CARRY=1', f'-DDENSE_DOWN_ROWS={X.FFN.DOWN_PC * 64}']
if BLOCK_CARRY and not PRODUCT_CORRECTION:
    raise ValueError('block carry requires product correction')
if BLOCK_CARRY:
    X.OS.append('-DGEMV_Q4_BLOCK_CARRY=1')
if PRODUCT_CORRECTION and not CORRECTION:
    raise ValueError('product correction requires corrected activation tables')
if PRODUCT_CORRECTION:
    X.OS.append('-DGEMV_Q4_PRODUCT_CORRECTION=1')
if CORRECTION:
    if not FULL or X.HID != 5120 or X.FF != 17408 or X.Q8:
        raise ValueError('corrected FFN probe requires H5120/FF17408 all-Q4 full FFN')
    # A corrected K8192 table would exceed L1. Reuse the exact pool and worker
    # with K4096 segments; up/gate K5120 now sets the maximum table size.
    X.FFN = replace(X.FFN, DOWN_SEGMENTS=segments(X.FF,4096))
    X.TAB_BYTES = projection_table_bytes(X.HID,X.TAB_BYTES,True)
    X.OS.append('-DGEMV_Q4_CORRECTION=1')
L = X.R.layout
OUT = L.A_OUT2
ACT_BYTES = max(L.A_BYTES, OUT + X.HID * 4 * len(X.FFN.DOWN_SEGMENTS))
if not X.FFN.DOWN_SEGMENTS or X.Q8:
    raise ValueError('segmented probe requires a segmented all-Q4 dense recipe')


@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def segmented(pool: In, act: InOut, trace: Out, *, source_hash: CompileTime[int] = 0):
    t = X.types()
    inc = include_dirs() + [str(HERE), str(HERE.parent / 'gemv_q4')]
    kernels = X.kernels(inc, t)
    wf = [ObjectFifo(t['elem'], name=f'w{c}', depth=2) for c in range(X.N_CORES)]
    xf = ObjectFifo(t['x'], name='x', depth=2)
    yf = [ObjectFifo(t['y'], name=f'y{c}', depth=2) for c in range(X.N_CORES)]

    tf = [ObjectFifo(t['y'], name=f't{c}', depth=2) for c in range(X.N_CORES)] if TRACE else []
    trace_fn = ExternalFunction('dense_trace', source_file=str(HERE / 'dense_trace.cc'),
                                arg_types=[t['ms'], t['y'], np.int32], include_dirs=inc) if TRACE else None

    def core_body(win, xin, yout, *args):
        buffers, functions = X.unpack_args(args[:-2] if TRACE else args)
        if TRACE:
            tout, copy = args[-2:]
            act_fn = functions['act']
            def traced_act(ms, y):
                offsets=X.range_(X.C.MS_U,X.C.MS_G+1,X.C.MS_G-X.C.MS_U) if ACTIVATION_SERIES else (X.C.MS_U,X.C.MS_G)
                for offset in offsets:
                    te = tout.acquire(1)
                    copy(ms, te, offset)
                    tout.release(1)
                act_fn(ms, y)
            functions['act'] = traced_act
        if FULL:
            X.ffn_body(win, xin, yout, buffers, functions)
        else:
            X.segmented_down_body(win, xin, yout, buffers, functions, DIAGNOSTIC)

    workers = [Worker(core_body,
                      fn_args=[wf[c].cons(), xf.cons(), yf[c].prod(),
                               *X.worker_args(X.core_buffers(t, c), kernels),
                               *([tf[c].prod(), trace_fn] if TRACE else [])],
                      tile=Tile(c, 2), stack_size=0x1800) for c in range(X.N_CORES)]
    wty = np.ndarray[(X.POOL_BYTES,), np.dtype[np.uint8]]
    aty = np.ndarray[(ACT_BYTES,), np.dtype[np.uint8]]

    trace_size = X.FF * 2 if TRACE else 1
    tty = np.ndarray[(trace_size,), np.dtype[np.float32]]

    def sequence(a_w, a_act, a_trace, w_prods, x_prod, y_conss, t_conss):
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        pt = Pipeline(3)
        if TRACE:
            for c in range(X.N_CORES):
                count = X.FFN.UP_PC * X.BAND_ROWS * 2
                pt.drain(t_conss[c], a_trace, X.bt(trace_size, c * count, count))
        if FULL:
            X.ffn_sequence(pw, px, py, a_w, a_act, w_prods, x_prod, y_conss,
                           ACT_BYTES, L.A_XM, L.A_H, OUT,
                           L.POOL_FFN_UP, L.POOL_FFN_GATE, L.POOL_FFN_DOWN)
        else:
            X.segmented_down_sequence(pw, px, py, a_w, a_act, w_prods, x_prod, y_conss,
                                      ACT_BYTES, L.A_H, OUT, L.POOL_FFN_DOWN, DIAGNOSTIC)
        pw.finish()
        px.finish()
        py.finish()
        pt.finish()

    rt = Runtime(sequence, [wty, aty, tty,
                            [f.prod(tile=Tile(c, 0)) for c, f in enumerate(wf)],
                            xf.prod(tile=Tile(1, 0)),
                            [f.cons(tile=Tile(c, 0)) for c, f in enumerate(yf)],
                            [f.cons(tile=Tile(c, 0)) for c, f in enumerate(tf)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = segmented
_sources = [Path(__file__), HERE / 'xcommon.py', HERE / 'gen_kernels.py', ROOT / 'ironutil.py',
            *sorted(HERE.glob('*.cc')), *sorted(HERE.glob('*.h')),
            *sorted((HERE.parent / 'gemv_q4').glob('*.h')),
            *sorted((ROOT / 'include').glob('*.h'))]
SPECIALIZE = {'source_hash': int(hashlib.sha256(b''.join(p.read_bytes() for p in _sources)
                         + b''.join(X.source_hash_inputs())
                         + repr((FULL, TRACE, DIAGNOSTIC, CORRECTION, PRODUCT_CORRECTION, BLOCK_CARRY, SEGMENT_CARRY, DOWN_RNE, ACTIVATION_CARRY, ACTIVATION_SERIES, X.C, X.FFN, L)).encode()).hexdigest()[:8], 16)}
