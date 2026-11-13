r"""gx: one GPT-OSS layer as two dispatches over dx.py's fabric (recipes/gptoss.py).

  GX_PART=att  ln -> q | k | v (+bias) -> YaRN RoPE -> attention with the learned sink -> o (+bias)
               -> res = x + o; xm = post_attention_norm(res)
               args: pool, xres f32[HID] (in), consts [lnw | postln | qn kn sinks], kv, act, ptab
  GX_PART=moe  per routed expert e: up | gate (+bias) -> w_e * clamped SwiGLU -> h_e; down over the
               four experts into each core's accumulator; xres = res + down + bsum
               args: pool, xres f32[HID] (out), act

The host routes between the two (core.cpp's gptoss_route): it reads xm, writes the four routing
weights beside it (act AD_W, inside xm's last x element) and the weighted down biases (AD_BSUM),
and patches this stream's expert fills, compiled as experts 0..3 (moeroute2).

dx.py's fabric: 8 main cores (Tile(c, 2)) on w (10 KB weight elements) / x (4 KB broadcast) /
y (64 floats) streams, the norm helper (Tile(0, 3)), and for attention 4 cores of 16 heads each
(Tile(2 + c, 3)) on one broadcast `ain`. The attention's GEMVs are gemv_q8.h's half-tile with the
bias in its pad (GEMV_Q8_PAD_BIAS); the experts' are gemv_mx4.h's (the GGUF's MXFP4, 9.25 KB w
elements of four half-tiles) unless the recipe keeps them at q8 (GX_MX4=0). The TUs come from
gen_kernels.py.

Build: OPEN_KERNELS_SPEC=<gptoss spec> GX_PART=att|moe python build_design.py designs/gptoss/gx.py <out>
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
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
ATTN = HERE.parent / "attn"
LN = HERE.parent / "ln"
LINL = HERE.parent / "lin_layer"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes import gptoss as GR  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM  # noqa: E402

SPEC = current_spec()
R = GR.recipe(SPEC)
L, G = R.layout, R.geo
PART = os.environ.get("GX_PART", "att")
assert PART in ("att", "moe"), f"GX_PART={PART!r} (att | moe)"
HID, QW, KVW, FF, HD = G.HID, G.QW, G.KVW, G.FF, G.HD
N_CORES, ACORES, NHL = G.N_CORES, G.ACORES, G.NHL
ELN, E_A = L.ELN, G.E_A
OS = ["-Os"]
PROBES = sorted(GR.probe_env().items())
ATTN_FLAGS = GR.attn_flags(G, [(k, v) for k, v in PROBES if k.startswith("ATTN_")])
LN_FLAGS = GR.ln_flags(G)
NULL = ["-DGEMV_NULL"] if os.environ.get("GX_NULL") else []
OGH = min(NHL, G.HPO)                  # heads per og element (attn.h kOGH)
N_OG = NHL // OGH
YB = BAND_ROWS * 4


def pb(K):
    """half-tiles per 64-row band"""
    return K // 64


def ng(K):
    """w elements per band"""
    return pb(K) // G.PER_CALL


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


BB_H, BB_Q, BB_F = GR.band_bytes(HID), GR.band_bytes(QW), GR.band_bytes(FF)
XBB_H, XBB_F = GR.band_bytes(HID, G.EXPERT_CH), GR.band_bytes(FF, G.EXPERT_CH)   # the experts' bands


def xng(K):
    """expert w elements per band"""
    return pb(K) // G.EXPERT_PER_CALL


def acq(fifo, n):
    e = fifo.acquire(n)
    return [e] if n == 1 else e


def gemv_bands(win, yout, tab, f_gy, nbands, ngroups, per_band):
    for _ in range_(nbands):
        ye = yout.acquire(1)
        for g in range_(ngroups):
            we = win.acquire(1)
            f_gy(we, tab, ye, g, per_band, 4)
            win.release(1)
        yout.release(1)


def _kernels(names, elem, x_ty, y_ty, tab_ty, ms_ty, acc_ty, wv_ty):
    i32 = np.int32
    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def ef(sym, args, flags=OS):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=flags)
    mk = {"gy": lambda: ef("gx_gy", [elem, tab_ty, y_ty, i32, i32, i32], OS + NULL),
          "gms": lambda: ef("gx_gms", [elem, tab_ty, ms_ty, i32, i32, i32], OS + NULL),
          "gacc": lambda: ef("gx_gacc", [elem, tab_ty, acc_ty, i32, i32, i32, i32], OS + NULL),
          "act": lambda: ef("gx_act", [ms_ty, y_ty, wv_ty, i32]),
          "wgrab": lambda: ef("gx_wgrab", [x_ty, wv_ty]),
          "copy": lambda: ef("gx_copy", [acc_ty, y_ty, i32]),
          "prep": lambda: ef("gx_prep", [x_ty, tab_ty, i32, i32]),
          "prepf": lambda: ef("gx_prepf", [x_ty, tab_ty, i32, i32])}
    return [mk[n]() for n in names]


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gx_att(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(G.CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(G.TAB_BYTES,), np.dtype[np.uint8]]
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(L.CD_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(L.KV_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(L.PTAB_BYTES,), np.dtype[np.uint8]]
    pb_ty = np.ndarray[(8,), np.dtype[np.int32]]              # [pos, nf, seen, -, blocks, remainder]
    bhd = np.ndarray[(HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    og_ty = np.ndarray[(OGH * HD,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(HD,), np.dtype[np.float32]]
    fq = np.ndarray[(2 * QW,), np.dtype[bfloat16]]          # ATTN_VEXP: q pre-split, [hi | lo]
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    fsk = np.ndarray[(NHL,), np.dtype[np.float32]]          # this core's sinks (attn.h ATTN_SINK)
    i32 = np.int32
    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def af(sym, src, args, flags):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    f_gy, f_prep = _kernels(("gy", "prep"), elem, x_ty, y_ty, tab_ty, None, None, None)
    f_nr = af("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4, LN_FLAGS)
    f_lny = af("ln_y", LN / "ln_y.cc", [u8_ln] * 5 + [i32], LN_FLAGS)
    f_lnx = af("ln_xn", LN / "ln_xn.cc", [u8_ln] * 6, LN_FLAGS)
    f_meta = af("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a, bhd, bhd, fcs, pb_ty, fsk, i32], ATTN_FLAGS)
    f_q = af("attn_q", ATTN / "attn_q.cc", [u8_a, bhd, fcs, fq, i32], ATTN_FLAGS)
    f_k = af("attn_k", ATTN / "attn_k.cc", [u8_a, bhd, fcs, fhd, brow, i32], ATTN_FLAGS)
    f_v = af("attn_v", ATTN / "attn_v.cc", [u8_a, brow, i32], ATTN_FLAGS)
    f_init = af("attn_init", ATTN / "attn_init.cc", [foacc, fml, fsk], ATTN_FLAGS)
    f_step = af("attn_step", ATTN / "attn_step.cc", [u8_a, u8_a, fq, foacc, fml, pb_ty, i32], ATTN_FLAGS)
    f_stepb = af("attn_stepb", ATTN / "attn_stepb.cc", [u8_a] * (2 * G.RB) + [fq, foacc, fml, pb_ty, i32],
                 ATTN_FLAGS)
    f_stepn = af("attn_step_new", ATTN / "attn_step_new.cc", [brow, brow, fq, foacc, fml, i32], ATTN_FLAGS)
    f_fin = af("attn_fin_ng", ATTN / "attn_fin_ng.cc", [foacc, fml, og_ty, i32], ATTN_FLAGS)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=max(G.XN_ELEMS, G.OG_ELEMS))
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_ain = ObjectFifo(u8_a, name="ain", depth=max(4, 2 * G.RB + 2))   # a block is acquired at once
    of_aout = ObjectFifo(brow, name="aout", depth=2)
    of_og = [ObjectFifo(og_ty, name=f"og{c}", depth=2) for c in range(ACORES)]

    def main_body(win, xin, yout, tab, f_gy, f_prep):
        # q | k | v against xn (K = HID)
        xe = acq(xin, G.XN_ELEMS)
        for i in range(G.XN_ELEMS):
            f_prep(xe[i], tab, HID, i)
        gemv_bands(win, yout, tab, f_gy, G.Q_PC + 2 * G.KV_PC, ng(HID), pb(HID))
        xin.release(G.XN_ELEMS)
        # o against og (K = QW)
        oe = acq(xin, G.OG_ELEMS)
        for i in range(G.OG_ELEMS):
            f_prep(oe[i], tab, QW, i)
        gemv_bands(win, yout, tab, f_gy, G.O_PC, ng(QW), pb(QW))
        xin.release(G.OG_ELEMS)

    def ln_body(ain, aout, f_nr, f_lny, f_lnx):
        # 1. the layer-entry norm: [x0 x1 lnw] -> [xn]
        e = ain.acquire(3)
        o = aout.acquire(1)
        f_nr(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)
        # 2. [x0 x1 postln out0 out1] -> [res0] [res1] [xm]: res = x + out, xm = post_attention_norm(res)
        e = ain.acquire(5)
        for i in range(2):
            o = aout.acquire(1)
            f_lny(e[0], e[1], e[3], e[4], o, i)
            aout.release(1)
        o = aout.acquire(1)
        f_lnx(e[0], e[1], e[3], e[4], e[2], o)
        aout.release(1)
        ain.release(5)

    def attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk, h0,
             f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
        e = ain.acquire(2)                                     # [qn | kn | sinks], the position record
        f_meta(e[0], e[1], qn, kn, cs, pbuf, sk, h0)
        ain.release(2)
        for h in range_(G.Q_AIN_ELEMS):
            e = ain.acquire(1)
            f_q(e, qn, cs, qs, h)
            ain.release(1)
        for h in range_(G.K_AIN_ELEMS):
            e = ain.acquire(1)
            f_k(e, kn, cs, tmp, kout, h)
            ain.release(1)
        for h in range_(G.K_AIN_ELEMS):
            e = ain.acquire(1)
            f_v(e, vout, h)
            ain.release(1)
        if aout is not None:                                   # core 0 owns the cache row
            o = aout.acquire(1)
            for j in range_(KVW):
                o[j] = kout[j]
            aout.release(1)
            o = aout.acquire(1)
            for j in range_(KVW):
                o[j] = vout[j]
            aout.release(1)
        f_init(oacc, ml, sk)
        for _ in range_(pbuf[4]):                              # whole blocks of RB cached rows
            e = ain.acquire(2 * G.RB)
            f_stepb(*[e[i] for i in range(2 * G.RB)], qs, oacc, ml, pbuf, h0)
            ain.release(2 * G.RB)
        for _ in range_(pbuf[5]):                              # what did not fill a block
            e = ain.acquire(2)
            f_step(e[0], e[1], qs, oacc, ml, pbuf, h0)
            ain.release(2)
        f_stepn(kout, vout, qs, oacc, ml, h0)
        for hp in range_(N_OG):
            o = ogout.acquire(1)
            f_fin(oacc, ml, o, hp)
            ogout.release(1)

    def attn0(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk,
              f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
        attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk, 0,
             f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb)

    def make_attn(c):
        def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk,
                 f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
            attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk, c * NHL,
                 f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb)
        return body

    workers = [Worker(ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_lny, f_lnx], tile=Tile(0, 3),
                      stack_size=GR.STACK)]
    for c in range(N_CORES):
        workers.append(Worker(main_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                   Buffer(tab_ty, name=f"tab{c}"), f_gy, f_prep],
                              tile=Tile(c, 2), stack_size=GR.STACK))

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s}"), Buffer(bhd, name=f"kn{s}"), Buffer(fcs, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(fhd, name=f"tmp{s}"), Buffer(brow, name=f"kout{s}"),
                Buffer(brow, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}"), Buffer(fsk, name=f"sk{s}")]
    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb]
    workers.append(Worker(attn0, fn_args=[of_ain.cons(), of_aout.prod(), of_og[0].prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=GR.STACK))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn(c), fn_args=[of_ain.cons(), of_og[c].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=GR.STACK))

    def sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c,
                 og_cs):
        # 1. layer-entry norm: xn -> act
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XN, ELN), wait=True, group=tg_ln)
        # 2. q | k | v: weights now, xn after the norm
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_Q + c * G.Q_PC * BB_H, G.Q_PC * BB_H))
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_K + c * G.KV_PC * BB_H, G.KV_PC * BB_H))
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_V + c * G.KV_PC * BB_H, G.KV_PC * BB_H))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_Q + c * G.Q_PC * YB, G.Q_PC * YB))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_KVN + c * G.KV_PC * YB, G.KV_PC * YB))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_KVN + KVW * 4 + c * G.KV_PC * YB, G.KV_PC * YB))
        tg_ln.finish()                                            # xn is in DDR
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_XN, G.XN_ELEMS * ELEM), wait=True, group=tg_x)
        # 3. attention: meta + record now, q / k / v after the GEMVs, the window, the new row out
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        pa_out.drain(aout_c, a_kv, bt(L.KV_BYTES, L.KV_ROW, L.KV_ROW))           # [k' | v'] -> row pos (attnpos)
        for c in range(ACORES):
            pa_out.drain(og_cs[c], a_act, bt(L.AD_BYTES, L.AD_OG + c * NHL * HD * 2, NHL * HD * 2))
        pa_in.fill(ain_p, a_consts, bt(L.CD_BYTES, L.CD_META, E_A))              # [qn | kn | sinks]
        pa_in.fill(ain_p, a_ptab, bt(L.PTAB_BYTES, L.PTAB_ROW, L.PTAB_ROW))      # the position record (attnpos)
        py.finish(*y_conss)                                       # q, k, v are in DDR
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_Q, QW * 4))
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN, KVW * 4))
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN + KVW * 4, KVW * 4))
        pa_in.fill(ain_p, a_kv, bt(L.KV_BYTES, 0, L.KV_ROW))                     # the window: rows [0, nf) (attnpos)
        # 4. o projection against og
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_O + c * G.O_PC * BB_Q, G.O_PC * BB_Q))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT + c * G.O_PC * YB, G.O_PC * YB))
        pa_out.finish()                                           # og (and the new cache row) are in DDR
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OG, G.OG_ELEMS * ELEM), wait=True, group=tg_x)
        # 5. res = x + out; xm = post_attention_norm(res)
        tg_ln2 = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTLN, ELN), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_RES, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XM, ELN), wait=True, group=tg_ln2)
        py.finish()                                               # out is in DDR
        lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT, HID * 4), wait=True, group=tg_ln2)
        tg_ln2.finish()
        pw.finish()
        pa_in.finish()
        tg_x.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0)),
                            [of_og[c].cons(tile=Tile(2 + c, 0)) for c in range(ACORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gx_moe(pool: In, xres: InOut, act: InOut, *, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(G.EXPERT_CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(G.TAB_BYTES,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(G.MS_FLOATS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(G.DOWN_PC * BAND_ROWS,), np.dtype[np.float32]]
    wv_ty = np.ndarray[(8,), np.dtype[np.float32]]
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]
    KN = ("gms", "gacc", "act", "wgrab", "copy", "prep", "prepf")
    KF = _kernels(KN, elem, x_ty, y_ty, tab_ty, ms_ty, acc_ty, wv_ty)
    f_add3 = ExternalFunction("ln_add3", source_file=str(LN / "ln_add3.cc"), arg_types=[u8_ln] * 4,
                              include_dirs=include_dirs(), compile_flags=LN_FLAGS)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=G.XM_ELEMS)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=3)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    NG_UG, PB_UG = xng(HID), pb(HID)
    NG_F, PB_F = xng(FF), pb(FF)

    def main_body(win, xin, yout, tab, ms, acc, wv, gms, gacc, act, wgrab, copy, prep, prepf):
        # xm and, in its last element, the routing weights
        xe = acq(xin, G.XM_ELEMS)
        for i in range(G.XM_ELEMS):
            prep(xe[i], tab, HID, i)
        wgrab(xe[G.XM_ELEMS - 1], wv)
        xin.release(G.XM_ELEMS)
        # per expert: up | gate per band -> h_e band (scaled by w_e)
        for e in range_(G.TOPK):
            for _ in range_(G.UP_PC):
                for g in range_(NG_UG):
                    we = win.acquire(1)
                    gms(we, tab, ms, g, PB_UG, G.MS_U)
                    win.release(1)
                for g in range_(NG_UG):
                    we = win.acquire(1)
                    gms(we, tab, ms, g, PB_UG, G.MS_G)
                    win.release(1)
                ye = yout.acquire(1)
                act(ms, ye, wv, e)
                yout.release(1)
        # down: h_e's table, then expert e's bands added onto the accumulator
        for e in range_(G.TOPK):
            for i in range_(G.H_ELEMS):
                he = xin.acquire(1)
                prepf(he, tab, FF, i)
                xin.release(1)
            for b in range_(G.DOWN_PC):
                for g in range_(NG_F):
                    we = win.acquire(1)
                    gacc(we, tab, acc, g, PB_F, b, e)
                    win.release(1)
        for b in range_(G.DOWN_PC):
            ye = yout.acquire(1)
            copy(acc, ye, b)
            yout.release(1)

    def ln_body(ain, aout, f_add3):
        # [res_i out2_i bsum_i] -> [xres_i], half by half
        for _ in range(2):
            e = ain.acquire(3)
            o = aout.acquire(1)
            f_add3(e[0], e[1], e[2], o)
            aout.release(1)
            ain.release(3)

    workers = [Worker(ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_add3], tile=Tile(0, 3), stack_size=GR.STACK)]
    for c in range(N_CORES):
        workers.append(Worker(main_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                   Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                   Buffer(acc_ty, name=f"acc{c}"), Buffer(wv_ty, name=f"wv{c}")] + KF,
                              tile=Tile(c, 2), stack_size=GR.STACK))

    def sequence(a_pool, c_xres, a_act, lni, lno, w_prods, x_prod, y_conss):
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        # 1. xm (+ the routing weights) to every main core
        px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XM, G.XM_ELEMS * ELEM))
        # 2. up | gate: each core's h bands of all four experts in ONE drain (expert e's at AD_H + e * FF * 4),
        #    then expert slot e's fills -- slot e is compiled as expert e, re-pointed by moeroute2
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, TensorAccessPattern((1, L.AD_BYTES), L.AD_H + c * G.UP_PC * YB,
                                                            [1, 1, G.TOPK, G.UP_PC * YB], [0, 0, FF * 4, 1]))
        for e in range(G.TOPK):
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, e * L.EXPERT_UG + c * G.UP_PC * 2 * XBB_H,
                                               G.UP_PC * 2 * XBB_H))
        py.finish()                                               # every h_e is in DDR
        # 3. down: h_e, then expert slot e's down bands; the cores hold the sum
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT2 + c * G.DOWN_PC * YB, G.DOWN_PC * YB))
        for e in range(G.TOPK):
            px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_H + e * FF * 4, G.H_ELEMS * ELEM))
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_DOWN + e * L.UP_BYTES + c * G.DOWN_PC * XBB_F,
                                               G.DOWN_PC * XBB_F))
        py.finish()                                               # out2 is in DDR
        # 4. xres = res + out2 + bsum
        tg = TaskGroup()
        for i in range(2):
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_RES + i * ELN, ELN), wait=True, group=tg)
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT2 + i * ELN, ELN), wait=True, group=tg)
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_BSUM + i * ELN, ELN), wait=True, group=tg)
        lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg)
        tg.finish()
        pw.finish()
        px.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, act_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = gx_att if PART == "att" else gx_moe
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + [(HERE.parent.parent / "recipes" / "gptoss.py").read_bytes(), (LN / "ln.h").read_bytes(),
                   (LN / "ln_y.cc").read_bytes(), (LN / "ln_xn.cc").read_bytes(), (LN / "ln_add3.cc").read_bytes(),
                   (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q8.h").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_mx4.h").read_bytes(), str(G.EXPERT_CH).encode(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode(), repr(PROBES).encode(), PART.encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
