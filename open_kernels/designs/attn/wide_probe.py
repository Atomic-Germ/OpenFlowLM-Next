"""Isolated production ax attention worker, exact recipe geometry, synthetic inputs.

Six BOs: meta u8[2*EA], q|gate f32[2*QW], k|v f32[2*KVW], interleaved
cache bf16[rows,2,KVW], new cache row bf16[2*KVW], gated output bf16[QW].
The position record must use nf=ATTN_PROBE_ROWS; masked future rows are legal.
This diagnostic does not export model support or change the legacy attn.py ABI.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from ironutil import Pipeline, include_dirs
from recipes.spec import ModelSpec
from recipes.qwen36moe import attn as geometry
from designs.attn import probe_support as P

FIXTURE = os.environ.get('ATTN_PROBE_FIXTURE', 'config_qwen38_27b.json')
SPEC = ModelSpec.from_hf_config(json.loads((ROOT.parent / 'specs/open-engine/tests/fixtures' / FIXTURE).read_text()))
D = geometry(SPEC)
ROWS = int(os.environ.get('ATTN_PROBE_ROWS', '257'))
if ROWS < 1 or D.RB != 1 or not D.VEXP or D.NHL != D.HPO:
    raise ValueError('attention probe requires rows>=1, VEXP, RB1 and one whole output element per core')
BODY = P.worker(D, range_)
FLAGS = [f'-DATTN_{k}={v}' for k,v in dict(NH=D.NH, KVH=D.KVH, HD=D.HD, ROT=D.ROT,
                                          GATE=1, VEXP=D.VEXP, NHL=D.NHL).items()]


@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def probe(meta: In, qg: In, kvn: In, cache: In, new: Out, og: Out, *, key: CompileTime[int]):
    def ty(n, dtype): return np.ndarray[(n,), np.dtype[dtype]]
    elem, row = ty(D.E_A, np.uint8), ty(D.KVW, bfloat16)
    head, cs, tmp = ty(D.HD, bfloat16), ty(D.ROT, np.float32), ty(D.HD, np.float32)
    qs, acc = ty(2*D.QW, bfloat16), ty(D.NHL*D.HD, np.float32)
    ml, pb = ty(2*D.MLS, np.float32), ty(4, np.int32)
    def fn(name, args):
        return ExternalFunction(name, source_file=str(HERE / f'{name}.cc'), arg_types=args,
                                include_dirs=include_dirs(), compile_flags=FLAGS)
    funcs = [fn('attn_meta', [elem,elem,head,head,cs,pb]),
             fn('attn_q', [elem,head,cs,qs,np.int32]),
             fn('attn_k', [elem,head,cs,tmp,row,np.int32]),
             fn('attn_v', [elem,row,np.int32]), fn('attn_init', [acc,ml]),
             fn('attn_step', [elem,elem,qs,acc,ml,pb,np.int32]),
             fn('attn_step_new', [row,row,qs,acc,ml,np.int32]),
             fn('attn_fin', [acc,ml,elem,elem,row,np.int32])]
    inp = ObjectFifo(elem, name='ain', depth=4)
    outs = [ObjectFifo(row, name=f'aout{c}', depth=2) for c in range(D.ACORES)]
    def make_body(c):
        def body(ain, out, qn, kn, cs, qs, tmp, kout, vout, acc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff):
            BODY(ain, out if c == 0 else None, out, qn, kn, cs, qs, tmp, kout, vout, acc, ml, pb,
                 fm, fq, fk, fv, fi, fs, fsn, ff, None, c)
        return body
    workers = []
    for c in range(D.ACORES):
        buffers = [Buffer(t, name=f'{name}{c}') for name,t in
                   [('qn',head),('kn',head),('cs',cs),('qs',qs),('tmp',tmp),('kout',row),
                    ('vout',row),('oacc',acc),('ml',ml),('pb',pb)]]
        workers.append(Worker(make_body(c), fn_args=[inp.cons(),outs[c].prod()]+buffers+funcs,
                              tile=Tile(c, 2), stack_size=0x1800))
    def sequence(a_meta, a_qg, a_kvn, a_cache, c_new, c_og, ip, ops):
        def tap(n,o,l): return TensorAccessPattern((1,n), o, [1,1,1,l], [0,0,0,1])
        P.sequence(D, ROWS, Pipeline(3), tap, a_meta,a_qg,a_kvn,a_cache,c_new,c_og,ip,ops)
    rt = Runtime(sequence, [ty(2*D.E_A,np.uint8),ty(2*D.QW,np.float32),ty(2*D.KVW,np.float32),
                            ty(ROWS*2*D.KVW,bfloat16),ty(2*D.KVW,bfloat16),ty(D.QW,bfloat16),
                            inp.prod(),[o.cons() for o in outs]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = probe
sources = [*HERE.glob('*.cc'), HERE/'attn.h', HERE/'probe_support.py', HERE/'wide_probe.py',
           P.AX, ROOT/'include/vecmath.h']
raw = b''.join(p.read_bytes() for p in sorted(sources)) + repr((D,ROWS)).encode()
SPECIALIZE = {'key': int(hashlib.sha256(raw).hexdigest()[:8],16)}
