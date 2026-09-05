r"""ax: a whole full-attention layer (attention block + MoE block) in ONE xclbin
context (phase 2 "whole-layer context"; the linear-layer twin is lx.py):

    ln -> gemv q | gate | k | v -> attn -> gemv o -> ln (+residual) -> router -> MoE

Same recipe as lx: 8 main cores (Tile(c, 2)) with the w / x / y streams run
every GEMV then the MoE; the ln + router core (Tile(0, 3)) and the attention
core (Tile(2, 3), attn.py's verbatim) are the helpers. Two instruction streams
on one xclbin (CompileTime `part`): 0 = everything up to the router, 1 = the
MoE (after the driver's `moeroute2`).

The cache position is NOT a build parameter (plan item 3, dynamic KV length):
the stream is built for the placeholder position 1 -- the KV window is ONE
linear fill of rows [0, nf) (the fifo delivers it as 2 nf 1 KB elements, K_t
V_t in order), the new row ONE 2 KB drain to row pos, the position record
(pos, nf, RoPE cos/sin) one 1 KB fill from ptab row pos -- and the driver's
`attnpos ax0 <pos>` rewrites those three words per token (the fill's BD
length, the two offsets). The attention core loops over the record's nf =
max(pos, 1) rows and masks rows t >= pos (position 0 streams one dummy row).

Args (layout.py): pool (q/k/v/gate/o and the experts at their pool offsets),
xres f32[2048] (in: layer input; out: layer output), consts [lnw | postln |
meta (qn | kn) | router W | sgw], kv (the layer's KV cache: MAX_CTX rows of
[K_t | V_t], updated in place), act (scratch), ptab (the position record
table, shared by the attention layers).
Build (WSL): for p in 0 1: AX_PART=$p python build_design.py designs/layer_x/ax.py designs/layer_x/build_ax$p
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (AA_BYTES, AA_HP, AA_KVN, AA_OG, AA_OUT, AA_QG, AA_RES, AA_ROUT, AA_XM, AA_XN,  # noqa: E402
                    CA_BYTES, CA_LNW, CA_META, CA_POSTLN, CA_RW, CA_SGW, KV_BYTES, KV_ROW, POOL_BYTES,
                    POOL_GATE, POOL_K, POOL_O, POOL_Q, POOL_V, PTAB_BYTES, PTAB_ROW)
import xcommon as X  # noqa: E402

HID = 2048
NH, KVH, HD = 16, 2, 256
N_CORES = 8
Q_PC, KV_PC, O_PC = 4096 // 64 // N_CORES, 512 // 64 // N_CORES, HID // 64 // N_CORES   # 8, 1, 4
PART = int(os.environ.get("AX_PART", 0))


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ax(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
       srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    u8_4k = tl["u8_4k"]
    u8_1k = np.ndarray[(1024,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CA_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(KV_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(AA_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]
    i32_4 = np.ndarray[(4,), np.dtype[np.int32]]
    b256 = np.ndarray[(HD,), np.dtype[bfloat16]]
    b512 = np.ndarray[(2 * HD,), np.dtype[bfloat16]]
    f64 = np.ndarray[(64,), np.dtype[np.float32]]
    f256 = np.ndarray[(HD,), np.dtype[np.float32]]
    f4096 = np.ndarray[(NH * HD,), np.dtype[np.float32]]
    f32_ = np.ndarray[(2 * NH,), np.dtype[np.float32]]

    inc = include_dirs() + [str(GEMV), str(ATTN), str(X.LN), str(X.LINL), str(X.RT), str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    f_meta = ExternalFunction("attn_meta", source_file=str(ATTN / "attn_meta.cc"), arg_types=[u8_1k, u8_1k, b256, b256, f64, i32_4], include_dirs=inc)
    f_q = ExternalFunction("attn_q", source_file=str(ATTN / "attn_q.cc"), arg_types=[u8_1k, b256, f64, f4096, np.int32], include_dirs=inc)
    f_k = ExternalFunction("attn_k", source_file=str(ATTN / "attn_k.cc"), arg_types=[u8_1k, b256, f64, f256, b512, np.int32], include_dirs=inc)
    f_v = ExternalFunction("attn_v", source_file=str(ATTN / "attn_v.cc"), arg_types=[u8_1k, b512, np.int32], include_dirs=inc)
    f_init = ExternalFunction("attn_init", source_file=str(ATTN / "attn_init.cc"), arg_types=[f4096, f32_], include_dirs=inc)
    f_step = ExternalFunction("attn_step", source_file=str(ATTN / "attn_step.cc"), arg_types=[u8_1k, u8_1k, f4096, f4096, f32_, i32_4], include_dirs=inc)
    f_stepn = ExternalFunction("attn_step_new", source_file=str(ATTN / "attn_step_new.cc"), arg_types=[b512, b512, f4096, f4096, f32_], include_dirs=inc)
    f_fin = ExternalFunction("attn_fin", source_file=str(ATTN / "attn_fin.cc"), arg_types=[f4096, f32_, u8_1k, u8_1k, b512, np.int32], include_dirs=inc)

    # ---- fifos
    of_w = [ObjectFifo(t["elem"], name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(t["y"], name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(t["x"], name="x", depth=2)
    of_lni = ObjectFifo(u8_4k, name="lni", depth=5)
    of_lno = ObjectFifo(u8_4k, name="lno", depth=3)
    of_ain = ObjectFifo(u8_1k, name="ain", depth=4)
    of_aout = ObjectFifo(b512, name="aout", depth=2)

    def main_body(win, xin, yout, *args):
        B, K = X.unpack_args(args)
        tab = B["tab"]
        xe = xin.acquire(1)                                     # xn
        K["prep2048"](xe, tab)
        X.gemv_bands(win, yout, tab, K["gy"], 2 * Q_PC + 2 * KV_PC, 8, 16, 2)   # q | gate | k | v, K = 2048
        xin.release(1)
        oe = xin.acquire(2)                                     # og, K = 4096
        K["prep4096a"](oe[0], tab)
        K["prep4096b"](oe[1], tab)
        X.gemv_bands(win, yout, tab, K["gy"], O_PC, 16, 32, 2)
        xin.release(2)
        X.moe_body(win, xin, yout, B, K)

    def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff):
        e = ain.acquire(2)                                      # [qn | kn], the position record
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
        for _ in range_(pb[1]):                                 # nf cached rows (K_t, V_t)
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

    workers = [Worker(X.ln_router_body,
                      fn_args=[of_lni.cons(), of_lno.prod(), Buffer(tl["xb"], name="rxs"), Buffer(tl["racc"], name="racc"),
                               L["ln_nr"], L["ln"], L["rcopy"], L["racc"], L["rfin"]],
                      tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body,
                              fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), *X.worker_args(X.core_buffers(t, c), K)],
                              tile=Tile(c, 2), stack_size=0x1800))
    workers.append(Worker(attn_body,
                          fn_args=[of_ain.cons(), of_aout.prod(),
                                   Buffer(b256, name="qn"), Buffer(b256, name="kn"), Buffer(f64, name="cs"),
                                   Buffer(f4096, name="qs"), Buffer(f256, name="tmp"), Buffer(b512, name="kout"),
                                   Buffer(b512, name="vout"), Buffer(f4096, name="oacc"), Buffer(f32_, name="ml"),
                                   Buffer(i32_4, name="pb"), f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin],
                          tile=Tile(2, 3), stack_size=0x1800))

    bt = X.bt
    BB16, BB32 = X.BAND16, X.BAND32

    def w_regions(c):
        return [(POOL_Q + c * Q_PC * BB16, Q_PC * BB16), (POOL_GATE + c * Q_PC * BB16, Q_PC * BB16),
                (POOL_K + c * KV_PC * BB16, KV_PC * BB16), (POOL_V + c * KV_PC * BB16, KV_PC * BB16),
                (POOL_O + c * O_PC * BB32, O_PC * BB32)]

    def y_regions(c):
        qb, kb = Q_PC * 64 * 4, KV_PC * 64 * 4
        return [(AA_QG + c * qb, qb), (AA_QG + NH * HD * 4 + c * qb, qb),
                (AA_KVN + c * kb, kb), (AA_KVN + KVH * HD * 4 + c * kb, kb),
                (AA_OUT + c * O_PC * 64 * 4, O_PC * 64 * 4)]

    def sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c):
        if part == 0:
            tg_ln = TaskGroup()
            lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
            lni.fill(a_consts, tap=bt(CA_BYTES, CA_LNW, 4096), wait=True, group=tg_ln)
            lno.drain(a_act, tap=bt(AA_BYTES, AA_XN, 4096), wait=True, group=tg_ln)
            pw, py = Pipeline(3), Pipeline(3)
            for c in range(N_CORES):
                for off, n in w_regions(c)[:3]:
                    pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, off, n))
                for off, n in y_regions(c)[:3]:
                    py.drain(y_conss[c], a_act, bt(AA_BYTES, off, n))
            tg_ln.finish()                                            # xn is in DDR
            tg_x = TaskGroup()
            x_prod.fill(a_act, tap=bt(AA_BYTES, AA_XN, 4096), wait=True, group=tg_x)
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[3]))
                py.drain(y_conss[c], a_act, bt(AA_BYTES, *y_regions(c)[3]))
            pa_out, pa_in = Pipeline(3), Pipeline(3)
            pa_out.drain(aout_c, a_kv, bt(KV_BYTES, KV_ROW, KV_ROW))        # the new row [k' | v'] -> row pos (attnpos)
            pa_out.drain(aout_c, a_act, bt(AA_BYTES, AA_OG, NH * HD * 2))
            pa_in.fill(ain_p, a_consts, bt(CA_BYTES, CA_META, 1024))        # [qn | kn]
            pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, PTAB_ROW, PTAB_ROW))   # the position record (attnpos)
            py.finish(*y_conss)                                       # q, gate, k, v are in DDR
            pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_QG, NH * HD * 4))
            pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_KVN, KVH * HD * 4))
            pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_KVN + KVH * HD * 4, KVH * HD * 4))
            pa_in.fill(ain_p, a_kv, bt(KV_BYTES, 0, KV_ROW))                 # the window: rows [0, nf) (attnpos)
            pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_QG + NH * HD * 4, NH * HD * 4))
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[4]))
                py.drain(y_conss[c], a_act, bt(AA_BYTES, *y_regions(c)[4]))
            tg_ln2 = TaskGroup()
            lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
            lni.fill(a_consts, tap=bt(CA_BYTES, CA_POSTLN, 4096), wait=True, group=tg_ln2)
            lno.drain(a_act, tap=bt(AA_BYTES, AA_RES, 8192), wait=True, group=tg_ln2)
            lno.drain(a_act, tap=bt(AA_BYTES, AA_XM, 4096), wait=True, group=tg_ln2)
            pa_out.finish()                                           # og (and the new cache rows) are in DDR
            x_prod.fill(a_act, tap=bt(AA_BYTES, AA_OG, 8192), wait=True, group=tg_x)
            py.finish()                                               # out is in DDR
            lni.fill(a_act, tap=bt(AA_BYTES, AA_OUT, HID * 4), wait=True, group=tg_ln2)
            tg_r = TaskGroup()
            lni.fill(a_consts, tap=bt(CA_BYTES, CA_RW, X.W_ELEMS * 4096), wait=True, group=tg_r)
            lno.drain(a_act, tap=bt(AA_BYTES, AA_ROUT, 4096), wait=True, group=tg_r)
            tg_ln2.finish()
            tg_r.finish()
            pw.finish()
            pa_in.finish()
            tg_x.finish()
        else:
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                           AA_BYTES, CA_BYTES, AA_XM, AA_ROUT, AA_RES, AA_HP, CA_SGW)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = ax
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + [(HERE / "xcommon.py").read_bytes()]
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
