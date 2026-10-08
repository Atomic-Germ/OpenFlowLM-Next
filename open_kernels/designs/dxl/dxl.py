r"""dxl: L consecutive positions through one dense layer in ONE dispatch, the weights
streamed once for all L (the verify / draft pass of speculative decoding).

    for each row: ln -> L-row gemv q | k | v -> attention, one row at a time (row j's KV
    reaches DDR before row j+1 reads its window) -> L-row gemv o -> for each row: ln
    (+residual) -> L-row gemv up | gate, silu(gate) * up -> L-row gemv down -> for each
    row: +residual

dx's fabric and buffers: 8 main cores (Tile(c, 2)), the ln core (Tile(0, 3)), the
attention cores (Tile(2 + c, 3)), and the same per-layer pool, consts, kv and ptab, so
a kernel set gains dxl without a second copy of any weight. Its own buffers are the
L-row residual `xres` f32[L][HID] and the L-row scratch `act` (recipes/dxl.py).

Per row the arithmetic is dx's -- the same ln and attention kernels, a GEMV that is
bit-identical per token to gemv_q4_tile (dxl_gemv.h), the same silu -- so the L rows'
logits are the ones L decode steps would give.

The attention patch sites are built for the placeholder positions 1 .. L (row j at
1 + j); the driver's `attnrows` patch moves them to pos0 + j.

Build: OPEN_KERNELS_SPEC=<spec> DXL_L=4 python build_design.py designs/dxl/dxl.py designs/dxl/build_<tag>
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
DESIGNS = HERE.parent
GEMV, ATTN, LN, LINL = DESIGNS / "gemv_q4", DESIGNS / "attn", DESIGNS / "ln", DESIGNS / "lin_layer"
sys.path.insert(0, str(DESIGNS.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes import dxl as DXR  # noqa: E402
from recipes.dxl import JMAX, NF, Z_ROW, job_table, jobs  # noqa: E402

SPEC = current_spec()
QR = for_spec(SPEC)
R = QR.recipe(SPEC)
L0, G = R.layout, R.geo                     # dx's layout: pool / consts / kv / ptab offsets
NL_ROWS = int(os.environ.get("DXL_L", 4))
X = DXR.layout(SPEC, NL_ROWS)
DRAFT = os.environ.get("DXL_DRAFT") == "1"    # which instruction stream: the cores and tables are the same
TABLE_INTS = -(-2 * (1 + JMAX * NF) // 16) * 16
LR = X.L
HID, FF, N_CORES = G.HID, G.FF, G.N_CORES
QW, KVW = G.QW, G.KVW
ELN, E_A = L0.ELN, L0.E_A
CHUNK = DXR.CHUNK
BB_H, BB_Q, BB_F = 2 * (HID // 256) * CHUNK, 2 * (QW // 256) * CHUNK, 2 * (FF // 256) * CHUNK
KSB, KSF = DXR.KS_BF16, DXR.KS_F32
YE = LR * 64                                # one band's [L][64] floats
BT, BP = X.BT, X.BP
YB = X.YB                                   # accumulator bands; the gate bands follow at YB
OS = ["-Os"]

ATTN_FLAGS = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
              f"-DATTN_QKNORM={1 if G.QKNORM else 0}", f"-DATTN_QKNORM_POST={1 if G.QKNORM_POST else 0}",
              f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}", f"-DATTN_NHL={G.NHL}"]
if G.RB > 1:
    ATTN_FLAGS.append(f"-DATTN_RB={G.RB}")
NPTAB, CSE = G.PTAB_ELEMS, G.PTAB_CS_ELEM
if NPTAB > 1:
    ATTN_FLAGS.append("-DATTN_PTAB_SPLIT=1")
for _k, _v in QR.probe_env().items():
    if _k not in ("ATTN_RB", "ATTN_FAST"):
        ATTN_FLAGS.append(f"-D{_k}={_v}")
ACORES, NHL, RB = G.ACORES, G.NHL, G.RB
OGH = min(NHL, G.HPO)
N_OG = NHL // OGH
LN_FLAGS = [f"-DLN_N={HID}", f"-DLN_EPS={G.EPS:g}f"]
if SPEC.norm_groups != 1:
    LN_FLAGS.append(f"-DLN_GROUPS={SPEC.norm_groups}")
GL_FLAGS = OS + [f"-DDXL_L={LR}", f"-DDXL_JMAX={JMAX}"]


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def tap(total, off, sizes, strides):
    return TensorAccessPattern((1, total), off, sizes, strides)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dxl(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, lora: In, *,
        srchash: CompileTime[int] = 0):
    elem = np.ndarray[(CHUNK,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(DXR.XE // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(YE,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(LR * DXR.tab_bytes(KSB),), np.dtype[np.uint8]]
    acc_ty = np.ndarray[((YB + BP) * YE,), np.dtype[np.float32]]
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L0.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(X.XRES_FLOATS,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(L0.CD_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(L0.KV_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(X.AD_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(L0.PTAB_BYTES,), np.dtype[np.uint8]]
    lora_ty = np.ndarray[(X.LORA_BYTES,), np.dtype[np.uint8]]
    # every core buffer a whole number of 64 B: the allocator packs them back to back, and a 512-bit
    # access at an address only 32 B aligned reads the wrong bytes without a fault
    table_ty = np.ndarray[(TABLE_INTS,), np.dtype[np.int32]]
    rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
    q4_ty = np.ndarray[(16,), np.dtype[np.int32]]
    jp_ty = np.ndarray[(16,), np.dtype[np.int32]]
    for ty in (elem, x_ty, y_ty, tab_ty, acc_ty, table_ty, rtp_ty, q4_ty, jp_ty):
        shape, dt = ty.__args__
        assert int(np.prod(shape[0] if isinstance(shape, tuple) else shape)) * np.dtype(dt.__args__[0]).itemsize % 64 == 0, ty
    pb_ty = np.ndarray[(8 if RB > 1 else 4,), np.dtype[np.int32]]
    bhd = np.ndarray[(G.HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    og_ty = np.ndarray[(OGH * G.HD,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(G.HD,), np.dtype[np.float32]]
    fq = (np.ndarray[(2 * QW,), np.dtype[bfloat16]] if G.VEXP else np.ndarray[(QW,), np.dtype[np.float32]])
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * G.HD,), np.dtype[np.float32]]
    i32 = np.int32

    inc = include_dirs() + [str(HERE), str(GEMV), str(ATTN), str(LN), str(LINL)]

    def ef(sym, src, args, flags=OS):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    f_njobs = ef("dxl_njobs", HERE / "dxl_njobs.cc", [table_ty, rtp_ty, q4_ty], GL_FLAGS)
    f_job = ef("dxl_job", HERE / "dxl_job.cc", [table_ty, rtp_ty, i32, jp_ty], GL_FLAGS)
    f_prep = ef("dxl_prep_job", HERE / "dxl_prep_job.cc", [x_ty, tab_ty, i32, jp_ty], GL_FLAGS)
    f_gemv = ef("dxl_gemv_job", HERE / "dxl_gemv_job.cc",
                [elem, tab_ty, acc_ty, jp_ty, i32, i32, i32], GL_FLAGS)
    f_out = ef("dxl_out_job", HERE / "dxl_out_job.cc", [acc_ty, y_ty, jp_ty, i32], GL_FLAGS)
    f_nr = ef("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4, LN_FLAGS)
    f_lny = ef("ln_y", LN / "ln_y.cc", [u8_ln] * 5 + [i32], LN_FLAGS)
    f_lnx = ef("ln_xn", LN / "ln_xn.cc", [u8_ln] * 6, LN_FLAGS)
    pz = [u8_a] if NPTAB > 1 else []
    f_meta = ef("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a] + pz + [bhd, bhd, fcs, pb_ty], ATTN_FLAGS)
    f_q = ef("attn_q", ATTN / "attn_q.cc", [u8_a, bhd, fcs, fq, i32], ATTN_FLAGS)
    f_k = ef("attn_k", ATTN / "attn_k.cc", [u8_a, bhd, fcs, fhd, brow, i32], ATTN_FLAGS)
    f_v = ef("attn_v", ATTN / "attn_v.cc", [u8_a, brow, i32], ATTN_FLAGS)
    f_init = ef("attn_init", ATTN / "attn_init.cc", [foacc, fml], ATTN_FLAGS)
    h0_arg = [i32] if ACORES > 1 else []
    f_step = ef("attn_step", ATTN / "attn_step.cc", [u8_a, u8_a, fq, foacc, fml, pb_ty] + h0_arg, ATTN_FLAGS)
    f_stepn = ef("attn_step_new", ATTN / "attn_step_new.cc", [brow, brow, fq, foacc, fml] + h0_arg, ATTN_FLAGS)
    f_stepb = (ef("attn_stepb", ATTN / "attn_stepb.cc", [u8_a] * (2 * RB) + [fq, foacc, fml, pb_ty] + h0_arg,
                  ATTN_FLAGS) if RB > 1 else None)
    f_fin = ef("attn_fin_ng", ATTN / "attn_fin_ng.cc", [foacc, fml, og_ty, i32], ATTN_FLAGS)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_ain = ObjectFifo(u8_a, name="ain", depth=max(4, 2 * RB + 2, 1 + NPTAB + 1))
    of_aout = ObjectFifo(brow, name="aout", depth=2)
    of_og = [ObjectFifo(og_ty, name=f"og{c}", depth=2) for c in range(ACORES)]

    tables = np.zeros(TABLE_INTS, np.int32)
    tables[:2 * (1 + JMAX * NF)] = job_table(SPEC, LR).reshape(-1)
    rtps = [Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(16, np.int32), use_write_rtp=True)
            for c in range(N_CORES)]
    barriers = [WorkerRuntimeBarrier() for _ in range(N_CORES)]

    def main_body(win, xin, yout, tab, acc, table, rtp, nj, jp, barrier, fnj, fjob, fprep, fgemv, fout):
        # the dispatch's mode is in rtp[0] before the barrier opens; nothing releases it, so the
        # next dispatch waits for its own stream to write its mode
        barrier.wait_for_value(1)
        fnj(table, rtp, nj)
        for j in range_(nj[0]):
            fjob(table, rtp, j, jp)                       # jp = the job's row; jp[0..3] its loop counts
            for s in range_(jp[0]):
                for tok in range_(LR):
                    xe = xin.acquire(1)
                    fprep(xe, tab, tok, jp)
                    xin.release(1)
                for b in range_(jp[1]):
                    for c in range_(jp[2]):
                        we = win.acquire(1)
                        fgemv(we, tab, acc, jp, b, c, s)
                        win.release(1)
            for b in range_(jp[3]):
                ye = yout.acquire(1)
                fout(acc, ye, jp, b)
                yout.release(1)

    def ln_body(ain, aout, f_nr, f_lny, f_lnx):
        for _ in range_(LR):                                     # entry norm, per row: [x0 x1 lnw] -> xn
            e = ain.acquire(3)
            o = aout.acquire(1)
            f_nr(e[0], e[1], e[2], o)
            aout.release(1)
            ain.release(3)

        def add_norm():
            # [x0 x1 w a0 a1] -> [y0] [y1] [xn]
            e = ain.acquire(5)
            for i in range(2):
                o = aout.acquire(1)
                f_lny(e[0], e[1], e[3], e[4], o, i)
                aout.release(1)
            o = aout.acquire(1)
            f_lnx(e[0], e[1], e[3], e[4], e[2], o)
            aout.release(1)
            ain.release(5)

        for _ in range_(LR):                                     # res = x + out; xm = norm(res)
            add_norm()
        for _ in range_(LR):                                     # xres = res + out2 (xn is junk)
            add_norm()

    def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
              f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, h0):
        for _ in range_(LR):                                     # one query row at a time
            e = ain.acquire(1 + NPTAB)
            if NPTAB > 1:
                f_meta(e[0], e[1], e[1 + CSE], qn, kn, cs, pb)
            else:
                f_meta(e[0], e[1], qn, kn, cs, pb)
            ain.release(1 + NPTAB)
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
            if aout is not None:
                o = aout.acquire(1)
                for j in range_(KVW):
                    o[j] = kout[j]
                aout.release(1)
                o = aout.acquire(1)
                for j in range_(KVW):
                    o[j] = vout[j]
                aout.release(1)
            f_init(oacc, ml)
            if RB > 1:
                for _ in range_(pb[4]):
                    e = ain.acquire(2 * RB)
                    args = [e[i] for i in range(2 * RB)] + [qs, oacc, ml, pb] + ([h0] if ACORES > 1 else [])
                    f_stepb(*args)
                    ain.release(2 * RB)
                for _ in range_(pb[5]):
                    e = ain.acquire(2)
                    f_step(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else f_step(e[0], e[1], qs, oacc, ml, pb)
                    ain.release(2)
            else:
                for _ in range_(pb[1]):
                    e = ain.acquire(2)
                    f_step(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else f_step(e[0], e[1], qs, oacc, ml, pb)
                    ain.release(2)
            f_stepn(kout, vout, qs, oacc, ml, h0) if ACORES > 1 else f_stepn(kout, vout, qs, oacc, ml)
            for hp in range_(N_OG):
                o = ogout.acquire(1)
                f_fin(oacc, ml, o, hp)
                ogout.release(1)

    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin] + ([f_stepb] if RB > 1 else [])

    def attn_body(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, *fns):
        fb = fns[8] if RB > 1 else None
        _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, *fns[:8], fb, 0)

    def make_attn_body(c):
        h0 = c * NHL

        def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, *fns):
            fb = fns[8] if RB > 1 else None
            _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, *fns[:8], fb, h0)
        return body

    workers = [Worker(ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_lny, f_lnx],
                      tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                  Buffer(tab_ty, name=f"tab{c}"), Buffer(acc_ty, name=f"acc{c}"),
                                                  Buffer(table_ty, name=f"jobs{c}", initial_value=tables),
                                                  rtps[c], Buffer(q4_ty, name=f"nj{c}"), Buffer(jp_ty, name=f"jp{c}"),
                                                  barriers[c], f_njobs, f_job, f_prep, f_gemv, f_out],
                              tile=Tile(c, 2), stack_size=0x1800))

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s}"), Buffer(bhd, name=f"kn{s}"), Buffer(fcs, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(fhd, name=f"tmp{s}"), Buffer(brow, name=f"kout{s}"),
                Buffer(brow, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]

    workers.append(Worker(attn_body, fn_args=[of_ain.cons(), of_aout.prod(), of_og[0].prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=0x1800))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn_body(c), fn_args=[of_ain.cons(), of_og[c].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=0x1800))

    def sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, a_lora, lni, lno, w_prods, x_prod, y_conss,
                 ain_p, aout_c, og_cs):
        AB, XF = X.AD_BYTES, X.XRES_FLOATS
        for c in range(N_CORES):
            rtps[c][0] = 1 if DRAFT else 0
        for c in range(N_CORES):
            barriers[c].set(1)
        pw, py, px, pl = Pipeline(3), Pipeline(3), Pipeline(3), Pipeline(3)
        stage = {}
        for job in jobs(SPEC, LR, DRAFT):
            stage.setdefault(job.stage, []).append(job)

        def w_fill(job):
            """core c's job-tile bands: [slices, bands, rows of 2 KB] of the pool or the LoRA pool"""
            k = job.core
            sb = k["CPS"] * CHUNK
            buf, size = (a_pool, L0.POOL_BYTES) if job.wbuf == "pool" else (a_lora, X.LORA_BYTES)
            for c in range(N_CORES):
                b0 = job.woff + (c * job.wpc + job.t * k["BT"]) * job.bb
                pw.fill(w_prods[c], buf, tap(size, b0, [k["S"], k["BT"], sb // 2048, 2048], [sb, job.bb, 2048, 1]))

        def x_fill(job):
            """the L rows' activation, slice-major; a LoRA tile's z rows, the seed row's from zero"""
            if job.x[0] == "z":
                px.fill(x_prod, a_act, bt(AB, X.AD_ZERO, DXR.XE))
                px.fill(x_prod, a_act, bt(AB, job.x[1] + Z_ROW, (LR - 1) * Z_ROW))
                return
            _, off, row, K, KS, f32 = job.x
            xb = KS * (4 if f32 else 2)
            px.fill(x_prod, a_act, tap(AB, off, [1, K // KS, LR, xb], [0, xb, row, 1]))

        def y_drain(job):
            """each core's [band][row][64] -> act[row][band*64 ..]"""
            if job.y is None:
                return
            off, row = job.y
            k = job.core
            for c in range(N_CORES):
                py.drain(y_conss[c], a_act, tap(AB, off + (c * job.wpc + job.t * k["BT"]) * 256,
                                                 [1, k["BT"], LR, 256], [0, 256, row, 1]))

        def run_stage(name, early=()):
            """a stage's jobs; the `early` ones had their weights and drains issued already.
            Before the first z fill, the A band's z must be in DDR."""
            zwait = True
            for job in stage[name]:
                # the wait comes before this job's own drain, which cannot finish without its z fill
                if job.x[0] == "z" and zwait:
                    py.finish()
                    zwait = False
                if job not in early:
                    y_drain(job)
                x_fill(job)
                if job not in early:
                    w_fill(job)

        def early(name):
            first = stage[name][0]
            y_drain(first)
            w_fill(first)
            return [first]

        # 1. entry norm, per row: xn_j -> act (the stage's first weights stream meanwhile)
        for j in range(LR):
            pl.fill(lni, c_xres, bt(XF, j * HID, HID))
            pl.fill(lni, a_consts, bt(L0.CD_BYTES, L0.CD_LNW, ELN))
            pl.drain(lno, a_act, bt(AB, X.AD_XN + j * X.XN_ROW, ELN))
        e = early("qkv")
        pl.finish()
        # 2. q | k | v
        run_stage("qkv", e)
        py.finish()                                              # q, k, v are in DDR
        # 3. attention, one row at a time: row j's KV row is in DDR before row j + 1's window
        e = []
        for j in range(LR):
            pa_out, pa_in = Pipeline(3), Pipeline(3)
            pa_out.drain(aout_c, a_kv, bt(L0.KV_BYTES, (1 + j) * L0.KV_ROW, L0.KV_ROW))    # attnrows
            for c in range(ACORES):
                pa_out.drain(og_cs[c], a_act, bt(AB, X.AD_OG + j * X.OG_ROW + c * NHL * G.HD * 2, NHL * G.HD * 2))
            pa_in.fill(ain_p, a_consts, bt(L0.CD_BYTES, L0.CD_META, E_A))
            pa_in.fill(ain_p, a_ptab, bt(L0.PTAB_BYTES, (1 + j) * L0.PTAB_ROW, L0.PTAB_ROW))  # attnrows
            pa_in.fill(ain_p, a_act, bt(AB, X.AD_Q + j * X.Q_ROW, QW * 4))
            pa_in.fill(ain_p, a_act, bt(AB, X.AD_K + j * X.KV_ROW_ACT, KVW * 4))
            pa_in.fill(ain_p, a_act, bt(AB, X.AD_V + j * X.KV_ROW_ACT, KVW * 4))
            pa_in.fill(ain_p, a_kv, bt(L0.KV_BYTES, 0, L0.KV_ROW))                          # attnrows window
            if j == 0:
                e = early("o")                                   # o's first weights stream under the attention
            pa_out.finish()
            pa_in.finish()
        # 4. o against og
        run_stage("o", e)
        py.finish()                                              # out is in DDR
        # 5. per row: res = x + out; xm = norm(res)
        for j in range(LR):
            pl.fill(lni, c_xres, bt(XF, j * HID, HID))
            pl.fill(lni, a_consts, bt(L0.CD_BYTES, L0.CD_POSTLN, ELN))
            pl.fill(lni, a_act, bt(AB, X.AD_OUT + j * X.HID_ROW, HID * 4))
            pl.drain(lno, a_act, bt(AB, X.AD_RES + j * X.HID_ROW, HID * 4))
            pl.drain(lno, a_act, bt(AB, X.AD_XM + j * X.XM_ROW, ELN))
        e = early("ug")
        pl.finish()                                              # res, xm are in DDR
        # 6. up | gate, silu -> h
        run_stage("ug", e)
        py.finish()                                              # h is in DDR
        # 7. down against h, then per row: xres = res + out2
        run_stage("down")
        py.finish()                                              # out2 is in DDR
        for j in range(LR):
            pl.fill(lni, a_act, bt(AB, X.AD_RES + j * X.HID_ROW, HID * 4))
            pl.fill(lni, a_consts, bt(L0.CD_BYTES, L0.CD_POSTLN, ELN))
            pl.fill(lni, a_act, bt(AB, X.AD_OUT2 + j * X.HID_ROW, HID * 4))
            pl.drain(lno, c_xres, bt(XF, j * HID, HID))
            pl.drain(lno, a_act, bt(AB, X.AD_JUNK, ELN))
        pl.finish()
        pw.finish()
        px.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty, lora_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0)),
                            [of_og[c].cons(tile=Tile(2 + c, 0)) for c in range(ACORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = dxl
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [Path(__file__).read_bytes()]
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + sorted(f.read_bytes() for f in (DESIGNS.parent / "recipes").glob("*.py"))
                + [(LN / "ln.h").read_bytes(), (LN / "ln_y.cc").read_bytes(), (LN / "ln_xn.cc").read_bytes(),
                   (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (DESIGNS.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode(), str(LR).encode(),
                   str(DRAFT).encode(), repr(GL_FLAGS).encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
