r"""attn_l: a full-attention decode layer as ONE dispatch (design A' of
.claude/plans/open-kernels-phase2-fused-linear-layer.md):

    ln -> gemv q | gate | k | v -> attn -> gemv o -> ln (+residual, post-attn norm)

Replaces 8 dispatches over 6 xclbin contexts. Same recipe as lin_layer: the
stages hand over through a DDR scratch BO (`act`), each dependent fill issued
after `dma_wait` on the drain that wrote it, weight streams and constants
issued ahead. Kernel bodies are gemv_q4's, attn's and ln's (+ ln_nr), unchanged.

Ten cores: one ln core does both norms (ln_nr then ln), the 8 GEMV cores stream
q (8 bands) | gate (8) | k (1) | v (1) against xn, then o (4 bands, K=4096)
against og -- two elements of one x fifo, bf16[4096] (xn's fill is 8 KB from
act[0], the tail unread) -- and the attention core is attn.py's. The cached-row
count `pos` is still a CompileTime parameter (plan item 3).

Shim budget (11 fills, 10 drains):
  col 0: w0 + ln in | y0 + ln out      col 2: w2 + attn in | y2 + attn out
  col 1: w1 + x     | y1               col 3-7: w | y
Per-channel start queues (4 BDs) are throttled by ironutil.Pipeline.

Args (see layout.py):
  w      u8[POOL_BYTES]  the layer pool (q/k/v/gate/o at their offsets)
  xres   f32[2048]       the layer's input residual
  consts u8[CA_BYTES]    [lnw | postln | meta]
  kv     bf16[KV]        the KV cache (K rows @0, V rows @KV_V_OFF, 1 KB per row)
  act    u8[AA_BYTES]    scratch: xn, qg, kvn, og, out; kvnew (the new cache rows) at AA_KVNEW
  hdr    u8[H_BYTES]     out: xm at 0, residual after attention at 12288 (the MoE header)
Build (WSL): ATTN_POS=11 python build_design.py designs/attn_layer/attn_l.py designs/attn_layer/build_pos11
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
ATTN = HERE.parent / "attn"
LN = HERE.parent / "ln"
LINL = HERE.parent / "lin_layer"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (AA_BYTES, AA_KVN, AA_KVNEW, AA_OG, AA_OUT, AA_QG, AA_XN, CA_BYTES, CA_LNW, CA_META, CA_POSTLN,  # noqa: E402
                    H_BYTES, H_XM, H_XRES, KV_BYTES, KV_V_OFF, POOL_BYTES, POOL_GATE, POOL_K, POOL_O, POOL_Q, POOL_V)

HID = 2048
NH, KVH, HD = 16, 2, 256
TILE_BYTES, PER_CALL, BAND_ROWS, N_CORES = 5120, 4, 64, 8
# q/gate/k/v: K=2048 -> 16-chunk bands (81920 B); o: K=4096 -> 32-chunk bands (163840 B)
PB16, PB32 = 16, 32
BB16, BB32 = PB16 * TILE_BYTES, PB32 * TILE_BYTES
CALL_BYTES = PER_CALL * TILE_BYTES
Q_PC, KV_PC, O_PC = 4096 // BAND_ROWS // N_CORES, 512 // BAND_ROWS // N_CORES, HID // BAND_ROWS // N_CORES   # 8, 1, 4
POS = int(os.environ.get("ATTN_POS", 0))
KV_ELEMS = KV_BYTES // 2
V_OFF = KV_V_OFF // 2


def bt(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def attn_l(w: In, xres: In, consts: In, kv: In, act: InOut, hdr: Out, *,
           pos: CompileTime[int], srchash: CompileTime[int] = 0):
    u8_4k = np.ndarray[(4096,), np.dtype[np.uint8]]
    u8_1k = np.ndarray[(1024,), np.dtype[np.uint8]]
    w_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CA_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(KV_ELEMS,), np.dtype[bfloat16]]
    act_ty = np.ndarray[(AA_BYTES,), np.dtype[np.uint8]]
    hdr_ty = np.ndarray[(H_BYTES,), np.dtype[np.uint8]]
    elem_ty = np.ndarray[(CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(2 * HID,), np.dtype[bfloat16]]      # og is bf16[4096]; xn uses the first half
    tab_ty = np.ndarray[(4 * HID + HID // 2,), np.dtype[np.uint8]]   # gemv_q4_tab_bytes(4096); K=2048 uses a prefix
    band_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    b256 = np.ndarray[(HD,), np.dtype[bfloat16]]
    b512 = np.ndarray[(2 * HD,), np.dtype[bfloat16]]
    f64 = np.ndarray[(64,), np.dtype[np.float32]]
    f256 = np.ndarray[(HD,), np.dtype[np.float32]]
    f4096 = np.ndarray[(NH * HD,), np.dtype[np.float32]]
    f32_ = np.ndarray[(2 * NH,), np.dtype[np.float32]]

    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]
    # ---- kernels
    ln_nr = ExternalFunction("ln_nr", source_file=str(LINL / "ln_nr.cc"), arg_types=[u8_4k] * 4, include_dirs=inc)
    ln_fn = ExternalFunction("ln_fn", source_file=str(LN / "ln.cc"), arg_types=[u8_4k] * 8, include_dirs=inc)
    ks16, ks32 = [], []
    for pb, ks in ((PB16, ks16), (PB32, ks32)):
        for i in range(pb // PER_CALL):
            src = GEMV / f"gemv_q4_p{PER_CALL}b{pb}r2_k{i}.cc"
            assert src.is_file(), f"{src} (generated by gemv_q4.py)"
            ks.append(ExternalFunction(src.stem, source_file=str(src), arg_types=[elem_ty, tab_ty, band_ty], include_dirs=inc))
    prep16 = ExternalFunction("gemv_q4_prep_k2048", source_file=str(GEMV / "gemv_q4_prep_k2048.cc"),
                              arg_types=[x_ty, tab_ty], include_dirs=inc)
    prep32 = ExternalFunction("gemv_q4_prep_k4096", source_file=str(GEMV / "gemv_q4_prep_k4096.cc"),
                              arg_types=[x_ty, tab_ty], include_dirs=inc)
    i32_4 = np.ndarray[(4,), np.dtype[np.int32]]
    f_meta = ExternalFunction("attn_meta", source_file=str(ATTN / "attn_meta.cc"), arg_types=[u8_1k, u8_1k, b256, b256, f64, i32_4], include_dirs=inc)
    f_q = ExternalFunction("attn_q", source_file=str(ATTN / "attn_q.cc"), arg_types=[u8_1k, b256, f64, f4096, np.int32], include_dirs=inc)
    f_k = ExternalFunction("attn_k", source_file=str(ATTN / "attn_k.cc"), arg_types=[u8_1k, b256, f64, f256, b512, np.int32], include_dirs=inc)
    f_v = ExternalFunction("attn_v", source_file=str(ATTN / "attn_v.cc"), arg_types=[u8_1k, b512, np.int32], include_dirs=inc)
    f_init = ExternalFunction("attn_init", source_file=str(ATTN / "attn_init.cc"), arg_types=[f4096, f32_], include_dirs=inc)
    f_step = ExternalFunction("attn_step", source_file=str(ATTN / "attn_step.cc"), arg_types=[u8_1k, u8_1k, f4096, f4096, f32_, i32_4], include_dirs=inc)
    f_stepn = ExternalFunction("attn_step_new", source_file=str(ATTN / "attn_step_new.cc"), arg_types=[b512, b512, f4096, f4096, f32_], include_dirs=inc)
    f_fin = ExternalFunction("attn_fin", source_file=str(ATTN / "attn_fin.cc"), arg_types=[f4096, f32_, u8_1k, u8_1k, b512, np.int32], include_dirs=inc)

    # ---- fifos
    of_lni = ObjectFifo(u8_4k, name="lni", depth=5)      # [x0 x1 w] then [x0 x1 w a0 a1]
    of_lno = ObjectFifo(u8_4k, name="lno", depth=3)      # [xn] then [y0 y1 xn]
    of_w = [ObjectFifo(elem_ty, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(band_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=1)
    of_ain = ObjectFifo(u8_1k, name="ain", depth=4)
    of_aout = ObjectFifo(b512, name="aout", depth=2)

    # ---- cores
    def ln_body(ain, aout, f1, f2):
        e = ain.acquire(3)
        o = aout.acquire(1)
        f1(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)
        e = ain.acquire(5)
        oo = aout.acquire(3)
        f2(e[0], e[1], e[3], e[4], e[2], oo[0], oo[1], oo[2])    # ln_fn(x0, x1, a0, a1, w, y0, y1, xn)
        aout.release(3)
        ain.release(5)

    def gemv_body(win, xin, yout, tab, fp16, fp32, *fns):
        f16, f32 = fns[:len(ks16)], fns[len(ks16):]
        xe = xin.acquire(1)                                     # xn
        fp16(xe, tab)                                           # block-quantise (gemv_q4.h), K = 2048
        for _ in range_(2 * Q_PC + 2 * KV_PC):                  # q | gate | k | v bands, K = 2048
            ye = yout.acquire(1)
            for fn in f16:
                we = win.acquire(1)
                fn(we, tab, ye)
                win.release(1)
            yout.release(1)
        xin.release(1)
        xe = xin.acquire(1)                                     # og
        fp32(xe, tab)                                           # K = 4096
        for _ in range_(O_PC):                                  # o bands, K = 4096
            ye = yout.acquire(1)
            for fn in f32:
                we = win.acquire(1)
                fn(we, tab, ye)
                win.release(1)
            yout.release(1)
        xin.release(1)

    def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff):
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
        for _ in range_(pb[1]):                                 # the record's nf = pos (static fills)
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

    workers = [Worker(ln_body, fn_args=[of_lni.cons(), of_lno.prod(), ln_nr, ln_fn], tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(gemv_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                  Buffer(tab_ty, name=f"tab{c}"), prep16, prep32, *ks16, *ks32],
                              tile=Tile(c, 2), stack_size=0x1000))
    workers.append(Worker(attn_body,
                          fn_args=[of_ain.cons(), of_aout.prod(),
                                   Buffer(b256, name="qn"), Buffer(b256, name="kn"), Buffer(f64, name="cs"),
                                   Buffer(f4096, name="qs"), Buffer(f256, name="tmp"), Buffer(b512, name="kout"),
                                   Buffer(b512, name="vout"), Buffer(f4096, name="oacc"), Buffer(f32_, name="ml"),
                                   Buffer(i32_4, name="pb"), f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin],
                          tile=Tile(2, 3), stack_size=0x1800))

    # per-core weight regions (pool offsets) and y placements (act byte offsets), in stream order
    def w_regions(c):
        return [(POOL_Q + c * Q_PC * BB16, Q_PC * BB16), (POOL_GATE + c * Q_PC * BB16, Q_PC * BB16),
                (POOL_K + c * KV_PC * BB16, KV_PC * BB16), (POOL_V + c * KV_PC * BB16, KV_PC * BB16),
                (POOL_O + c * O_PC * BB32, O_PC * BB32)]

    def y_regions(c):
        qb, kb = Q_PC * BAND_ROWS * 4, KV_PC * BAND_ROWS * 4
        return [(AA_QG + c * qb, qb), (AA_QG + NH * HD * 4 + c * qb, qb),
                (AA_KVN + c * kb, kb), (AA_KVN + KVH * HD * 4 + c * kb, kb),
                (AA_OUT + c * O_PC * BAND_ROWS * 4, O_PC * BAND_ROWS * 4)]

    # ---- host sequence
    def sequence(a_w, a_xres, a_consts, a_kv, a_act, c_hdr, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c):
        # 1. layer-entry norm: xn -> act[AA_XN]
        tg_ln = TaskGroup()
        lni.fill(a_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(CA_BYTES, CA_LNW, 4096), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(AA_BYTES, AA_XN, 4096), wait=True, group=tg_ln)
        # 2. q | gate | k | v GEMVs: three weight fills and three y drains queued per core (4-BD start
        #    queues), the rest after the norm has produced xn (the throttle waits on core progress)
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            for off, n in w_regions(c)[:3]:
                pw.fill(w_prods[c], a_w, bt(POOL_BYTES, off, n))
            for off, n in y_regions(c)[:3]:
                py.drain(y_conss[c], a_act, bt(AA_BYTES, off, n))
        tg_ln.finish()                                            # xn is in DDR
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(AA_BYTES, AA_XN, 2 * HID * 2), wait=True, group=tg_x)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_w, bt(POOL_BYTES, *w_regions(c)[3]))
            py.drain(y_conss[c], a_act, bt(AA_BYTES, *y_regions(c)[3]))
        # 3. attention: drains first, meta now, q/k/v/cache/gate once the projections have landed
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        pa_out.drain(aout_c, a_act, bt(AA_BYTES, AA_KVNEW, 2 * HD * 2))
        pa_out.drain(aout_c, a_act, bt(AA_BYTES, AA_KVNEW + 2 * HD * 2, 2 * HD * 2))
        pa_out.drain(aout_c, a_act, bt(AA_BYTES, AA_OG, NH * HD * 2))
        pa_in.fill(ain_p, a_consts, bt(CA_BYTES, CA_META, 2048))
        py.finish(*y_conss)                                       # q, gate, k, v are in DDR
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_QG, NH * HD * 4))
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_KVN, KVH * HD * 4))
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_KVN + KVH * HD * 4, KVH * HD * 4))
        for t in range(pos):
            pa_in.fill(ain_p, a_kv, bt(KV_ELEMS, t * 512, 512))
            pa_in.fill(ain_p, a_kv, bt(KV_ELEMS, V_OFF + t * 512, 512))
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_QG + NH * HD * 4, NH * HD * 4))
        # 4. o GEMV: weights now, og after the attention has landed
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_w, bt(POOL_BYTES, *w_regions(c)[4]))
            py.drain(y_conss[c], a_act, bt(AA_BYTES, *y_regions(c)[4]))
        tg_ln2 = TaskGroup()
        lni.fill(a_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(CA_BYTES, CA_POSTLN, 4096), wait=True, group=tg_ln2)
        lno.drain(c_hdr, tap=bt(H_BYTES, H_XRES, 8192), wait=True, group=tg_ln2)
        lno.drain(c_hdr, tap=bt(H_BYTES, H_XM, 4096), wait=True, group=tg_ln2)
        pa_out.finish()                                           # og (and the new cache rows) are in DDR
        x_prod.fill(a_act, tap=bt(AA_BYTES, AA_OG, NH * HD * 2), wait=True, group=tg_x)
        py.finish()                                               # out is in DDR
        # 5. residual + post-attention norm -> the MoE header
        lni.fill(a_act, tap=bt(AA_BYTES, AA_OUT, HID * 4), wait=True, group=tg_ln2)
        tg_ln2.finish()
        pw.finish()
        pa_in.finish()
        tg_x.finish()

    rt = Runtime(sequence, [w_ty, xres_ty, consts_ty, kv_ty, act_ty, hdr_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(2, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = attn_l
_src = b"".join(sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + [(LN / "ln.cc").read_bytes(), (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"pos": POS, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
