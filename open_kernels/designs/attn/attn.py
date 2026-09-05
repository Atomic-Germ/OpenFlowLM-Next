r"""Full-attention decode step on the NPU (one core): head norms + partial RoPE,
online-softmax attention over `pos` cached rows plus the new position, sigmoid
gate. Math in attn.h.

Args (6 -- the firmware rejects runs with ~9+ buffers): meta u8[2048] (attn.h's two
elements: [qn | kn] @0, the position record [pos | nf | cos @512 | sin @640] @1024),
qg f32[8192] ([q | gate], two GEMVs sharing one BO via GEMV_YOFF), kvn f32[1024]
([k | v]), kv bf16[1572864] (the 3 MB cache: K rows @0, V rows @byte 1073152,
1 KB per row), kvnew bf16[1024] (out: [k' | v'] cache rows), og bf16[4096] (out).
`pos` (rows already in the cache) is a CompileTime parameter of this standalone
design: its row fills are static, and the record's nf must equal it. The
whole-layer design layer_x/ax.py streams the window as one driver-patched fill.
Build: ATTN_POS=11 python build_design.py designs/attn/attn.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

NH, KVH, HD = 16, 2, 256
KV_ELEMS = 3145728 // 2
V_OFF = 1073152 // 2            # V rows start (bf16 elements)
POS = int(os.environ.get("ATTN_POS", 11))


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def attn(meta: In, qg: In, kvn: In, kv: In, kvnew: Out, og: Out, *,
         pos: CompileTime[int], srchash: CompileTime[int] = 0):
    u8 = np.ndarray[(1024,), np.dtype[np.uint8]]
    b256 = np.ndarray[(HD,), np.dtype[bfloat16]]
    b512 = np.ndarray[(2 * HD,), np.dtype[bfloat16]]
    f64 = np.ndarray[(64,), np.dtype[np.float32]]
    f256 = np.ndarray[(HD,), np.dtype[np.float32]]
    f4096 = np.ndarray[(NH * HD,), np.dtype[np.float32]]
    f32_ = np.ndarray[(2 * NH,), np.dtype[np.float32]]
    meta_ty = np.ndarray[(2048,), np.dtype[np.uint8]]
    qg_ty = np.ndarray[(2 * NH * HD,), np.dtype[np.float32]]      # [q | gate]
    kvn_ty = np.ndarray[(2 * KVH * HD,), np.dtype[np.float32]]    # [k | v]
    kvnew_ty = np.ndarray[(2 * 2 * HD,), np.dtype[bfloat16]]      # [k' | v'] cache rows
    cache_ty = np.ndarray[(KV_ELEMS,), np.dtype[bfloat16]]
    og_ty = np.ndarray[(NH * HD,), np.dtype[bfloat16]]
    inc = include_dirs()
    i32_4 = np.ndarray[(4,), np.dtype[np.int32]]
    f_meta = ExternalFunction("attn_meta", source_file=str(HERE / "attn_meta.cc"), arg_types=[u8, u8, b256, b256, f64, i32_4], include_dirs=inc)
    f_q = ExternalFunction("attn_q", source_file=str(HERE / "attn_q.cc"), arg_types=[u8, b256, f64, f4096, np.int32], include_dirs=inc)
    f_k = ExternalFunction("attn_k", source_file=str(HERE / "attn_k.cc"), arg_types=[u8, b256, f64, f256, b512, np.int32], include_dirs=inc)
    f_v = ExternalFunction("attn_v", source_file=str(HERE / "attn_v.cc"), arg_types=[u8, b512, np.int32], include_dirs=inc)
    f_init = ExternalFunction("attn_init", source_file=str(HERE / "attn_init.cc"), arg_types=[f4096, f32_], include_dirs=inc)
    f_step = ExternalFunction("attn_step", source_file=str(HERE / "attn_step.cc"), arg_types=[u8, u8, f4096, f4096, f32_, i32_4], include_dirs=inc)
    f_stepn = ExternalFunction("attn_step_new", source_file=str(HERE / "attn_step_new.cc"), arg_types=[b512, b512, f4096, f4096, f32_], include_dirs=inc)
    f_fin = ExternalFunction("attn_fin", source_file=str(HERE / "attn_fin.cc"), arg_types=[f4096, f32_, u8, u8, b512, np.int32], include_dirs=inc)

    of_in = ObjectFifo(u8, name="in", depth=4)
    of_out = ObjectFifo(b512, name="out", depth=2)   # bf16 elements: knew, vnew, og (8)
    qn = Buffer(b256, name="qn"); kn = Buffer(b256, name="kn"); cs = Buffer(f64, name="cs")
    qs = Buffer(f4096, name="qs"); tmp = Buffer(f256, name="tmp")
    kout = Buffer(b512, name="kout"); vout = Buffer(b512, name="vout")
    oacc = Buffer(f4096, name="oacc"); ml = Buffer(f32_, name="ml"); pb = Buffer(i32_4, name="pb")

    def core_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff):
        e = ain.acquire(2)
        fm(e[0], e[1], qn, kn, cs, pb)
        ain.release(2)
        for h in range_(NH):
            e = ain.acquire(1)
            fq(e, qn, cs, qs, h)
            ain.release(1)
        for h in range_(KVH):
            e = ain.acquire(1)
            fk(e, kn, cs, tmp, kout, h)
            ain.release(1)
        for h in range_(KVH):
            e = ain.acquire(1)
            fv(e, vout, h)
            ain.release(1)
        o = aout.acquire(1)
        for j in range_(2 * HD):
            o[j] = kout[j]
        aout.release(1)
        o = aout.acquire(1)
        for j in range_(2 * HD):
            o[j] = vout[j]
        aout.release(1)
        fi(oacc, ml)
        for _ in range_(pb[1]):
            e = ain.acquire(2)
            fs(e[0], e[1], qs, oacc, ml, pb)
            ain.release(2)
        fsn(kout, vout, qs, oacc, ml)
        for hp in range_(NH // 2):
            g = ain.acquire(2)
            o = aout.acquire(1)
            ff(oacc, ml, g[0], g[1], o, hp)
            aout.release(1)
            ain.release(2)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                                        f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_meta, a_qg, a_kvn, a_kv, c_kvnew, c_og, inp, outc):
        pipe = Pipeline(3)
        pipe.drain(outc, c_kvnew, TensorAccessPattern((1, 4 * HD), 0, [1, 1, 1, 2 * HD], [0, 0, 0, 1]))
        pipe.drain(outc, c_kvnew, TensorAccessPattern((1, 4 * HD), 2 * HD, [1, 1, 1, 2 * HD], [0, 0, 0, 1]))
        pipe.drain(outc, c_og, TensorAccessPattern((1, NH * HD), 0, [1, 1, 1, NH * HD], [0, 0, 0, 1]))
        pipe.fill(inp, a_meta, TensorAccessPattern((1, 2048), 0, [1, 1, 1, 2048], [0, 0, 0, 1]))
        pipe.fill(inp, a_qg, TensorAccessPattern((1, 2 * NH * HD), 0, [1, 1, 1, NH * HD], [0, 0, 0, 1]))
        pipe.fill(inp, a_kvn, TensorAccessPattern((1, 2 * KVH * HD), 0, [1, 1, 1, KVH * HD], [0, 0, 0, 1]))
        pipe.fill(inp, a_kvn, TensorAccessPattern((1, 2 * KVH * HD), KVH * HD, [1, 1, 1, KVH * HD], [0, 0, 0, 1]))
        for t in range(pos):
            pipe.fill(inp, a_kv, TensorAccessPattern((1, KV_ELEMS), t * 512, [1, 1, 1, 512], [0, 0, 0, 1]))
            pipe.fill(inp, a_kv, TensorAccessPattern((1, KV_ELEMS), V_OFF + t * 512, [1, 1, 1, 512], [0, 0, 0, 1]))
        pipe.fill(inp, a_qg, TensorAccessPattern((1, 2 * NH * HD), NH * HD, [1, 1, 1, NH * HD], [0, 0, 0, 1]))
        pipe.finish()

    rt = Runtime(sequence, [meta_ty, qg_ty, kvn_ty, cache_ty, kvnew_ty, og_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = attn
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h")) + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"pos": POS, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
