r"""sx: a whole LFM2-MoE short-conv layer (conv mixer + MoE block) in ONE xclbin
context (the lx / ax twin for the short-conv mixer):

    ln -> gemv B | C | u -> short conv (helper) -> gemv out -> ln (+residual) -> router -> MoE

Like lx: 8 main cores (Tile(c, 2)) run every GEMV and the MoE in one core
program fed by w / x / y streams. Helpers: ln + router (Tile(0, 3)) and the
short-conv core (Tile(2, 3), short_conv/sc.h). Two instruction streams (part):
0 = everything up to the router, 1 = the MoE (after the driver's moeroute2).

Args: pool u8[POOL_BYTES] (B, C, u, out and the experts at their pool offsets),
xres f32[HID] (in: layer input; out: layer output), consts (per layer), state
((taps-1) x hidden f32, in place), act (scratch).
Build: for p in 0 1: SX_PART=$p python build_design.py designs/layer_x/sx.py designs/layer_x/build_sx$p
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
SC = HERE.parent / "short_conv"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (AA_BYTES, AA_OUT, AA_ROUT, AA_XM, AA_XN, AA_RES, AA_HP, AA_QG, AA_KVN,
                    CA_BYTES, CA_BIAS, CA_LNW, CA_POSTLN, CA_RW, CA_SGW, ELN, POOL_BYTES,
                    POOL_SC_B, POOL_SC_C, POOL_SC_U, POOL_SC_OUT, STATE_BYTES,
                    CD_CONV, CD_BYTES, AD_B, AD_C, AD_U, AD_Y, AD_BYTES, R, SPEC)
import xcommon as X  # noqa: E402
import xlayer as XL  # noqa: E402

from recipes import lfm2moe as FAM  # noqa: E402

HID = X.HID
N_CORES = X.N_CORES
ELEM = X.ELEM
TAPS = FAM.TAPS
SC_PC = HID // (X.BAND_ROWS * N_CORES)   # bands per core for a [hidden, hidden] projection
SCW = 512
PART = int(os.environ.get("SX_PART", 0))
STOP = int(os.environ.get("SX_STOP", 99))

@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def sx(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, *, part: CompileTime[int] = 0,
       stop: CompileTime[int] = 99, srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CD_BYTES,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(AD_BYTES,), np.dtype[np.uint8]]
    fsc = np.ndarray[(SCW,), np.dtype[np.float32]]
    bsc = np.ndarray[(SCW,), np.dtype[bfloat16]]

    inc = include_dirs() + [str(GEMV), str(SC), str(X.LN), str(X.RT), str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    from aie.iron.kernel import ExternalFunction
    f_sc = ExternalFunction("short_conv_step", source_file=str(SC / "sc_elem.cc"),
                            arg_types=[fsc, fsc, fsc, bsc, bsc, bsc, fsc, fsc, bsc, fsc, fsc],
                            include_dirs=inc, compile_flags=[f"-DSC_TAPS={TAPS}", f"-DSC_W={SCW}", "-DSC_PASSTHROUGH=0"])

    of_w, of_y, of_x = XL.main_fifos(t)
    of_lni, of_lno = XL.ln_fifos(tl)
    from aie.iron import ObjectFifo
    of_scin = ObjectFifo(fsc, name="scin", depth=6)
    of_scw = ObjectFifo(bsc, name="scw", depth=4)
    of_scy = ObjectFifo(bsc, name="scy", depth=4)
    of_scst = ObjectFifo(fsc, name="scst", depth=4)

    def main_body(win, xin, yout, *args):
        B, K = X.unpack_args(args)
        tab = B["tab"]
        xe = xin.acquire(1)
        K["prep2048"](xe, tab)
        X.role_gemv_bands(win, yout, B, K, "linear", 3 * SC_PC, HID)
        xin.release(1)
        oe = xin.acquire(1)
        K["prep2048"](oe, tab)
        X.role_gemv_bands(win, yout, B, K, "linear_out", SC_PC, HID)
        xin.release(1)
        X.moe_body(win, xin, yout, B, K)

    def sc_body(scin, scw, scy, scst, f_sc):
        for _ in range_(HID // SCW):
            w = scw.acquire(TAPS)
            e = scin.acquire(5)
            oy = scy.acquire(1)
            os = scst.acquire(2)
            f_sc(e[0], e[1], e[2], w[0], w[1], w[2], e[3], e[4], oy, os[0], os[1])
            scst.release(2)
            scy.release(1)
            scin.release(5)
            scw.release(TAPS)

    workers = ([XL.ln_worker(of_lni, of_lno, tl, L)] + XL.main_workers(main_body, of_w, of_x, of_y, t, K)
               + [Worker(sc_body, fn_args=[of_scin.cons(), of_scw.cons(), of_scy.prod(), of_scst.prod(), f_sc],
                         tile=Tile(2, 3), stack_size=0x1800)])

    def sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                 scin_p, scw_p, scy_c, scst_c):
        bt = X.bt
        if part == 1:
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                           AA_BYTES, CA_BYTES, AA_XM, AA_ROUT, AA_RES, AA_HP, CA_SGW)
            return
        BB_HID = X.role_band_bytes("linear", HID)
        BB_OUT = X.role_band_bytes("linear_out", HID)
        YB = X.BAND_ROWS * 4
        SCB = 512 * 4
        SCYB = 512 * 2
        TAPB = 512 * 2
        # 1. entry norm
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(CD_BYTES, CA_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(AD_BYTES, AA_XN, ELN), wait=True, group=tg_ln)
        tg_ln.finish()
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(AD_BYTES, AA_XN, ELEM), wait=True, group=tg_x)
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            for base in (POOL_SC_B, POOL_SC_C, POOL_SC_U):
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, base + c * SC_PC * BB_HID, SC_PC * BB_HID))
            for dst in (AD_B, AD_C, AD_U):
                py.drain(y_conss[c], a_act, bt(AD_BYTES, dst + c * SC_PC * YB, SC_PC * YB))
        # 3. conv core
        ps_in, ps_out = Pipeline(3), Pipeline(3)
        for i in range(HID // SCW):
            for k in range(TAPS):
                ps_in.fill(scw_p, a_consts, bt(CD_BYTES, CD_CONV + (k * HID + i * SCW) * 2, TAPB))
        py.finish(*y_conss)
        for i in range(HID // SCW):
            ps_in.fill(scin_p, a_act, bt(AD_BYTES, AD_B + i * SCB, SCB))
            ps_in.fill(scin_p, a_act, bt(AD_BYTES, AD_C + i * SCB, SCB))
            ps_in.fill(scin_p, a_act, bt(AD_BYTES, AD_U + i * SCB, SCB))
            ps_in.fill(scin_p, a_state, bt(STATE_BYTES, i * SCB, SCB))
            ps_in.fill(scin_p, a_state, bt(STATE_BYTES, HID * 4 + i * SCB, SCB))
            ps_out.drain(scy_c, a_act, bt(AD_BYTES, AD_Y + i * SCYB, SCYB))
            ps_out.drain(scst_c, a_state, bt(STATE_BYTES, i * SCB, SCB))
            ps_out.drain(scst_c, a_state, bt(STATE_BYTES, HID * 4 + i * SCB, SCB))
        ps_out.finish()
        ps_in.finish()
        # 4. out
        tg_o = TaskGroup()
        x_prod.fill(a_act, tap=bt(AD_BYTES, AD_Y, ELEM), wait=True, group=tg_o)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_SC_OUT + c * SC_PC * BB_HID, SC_PC * BB_HID))
            py.drain(y_conss[c], a_act, bt(AD_BYTES, AA_OUT + c * SC_PC * YB, SC_PC * YB))
        tg_o.finish()
        targ = [c for c in range(N_CORES)]
        # 5. residual + norm + router (the full-sequence tail)
        tg_ln2 = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(CD_BYTES, CA_POSTLN, ELN), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(AD_BYTES, AA_RES, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(AD_BYTES, AA_XM, ELN), wait=True, group=tg_ln2)
        if stop == 2:
            py.finish()
            pw.finish()
            tg_x.finish()
            return
        py.finish()
        lni.fill(a_act, tap=bt(AD_BYTES, AA_OUT, HID * 4), wait=True, group=tg_ln2)
        tg_r = TaskGroup()
        lni.fill(a_consts, tap=bt(CD_BYTES, CA_RW, X.W_ELEMS * ELEM), wait=True, group=tg_r)
        lni.fill(a_consts, tap=bt(CD_BYTES, CA_BIAS, ELEM), wait=True, group=tg_r)
        lno.drain(a_act, tap=bt(AD_BYTES, AA_ROUT, ELEM), wait=True, group=tg_r)
        tg_ln2.finish()
        tg_r.finish()
        pw.finish()
        tg_x.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_scin.prod(tile=Tile(2, 0)), of_scw.prod(tile=Tile(3, 0)),
                            of_scy.cons(tile=Tile(1, 0)), of_scst.cons(tile=Tile(2, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = sx
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes(), (HERE / "xlayer.py").read_bytes(), (HERE / "sx.py").read_bytes()]
                + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in SC.glob("*.cc")) + sorted(f.read_bytes() for f in SC.glob("*.h"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(),
                   (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "stop": STOP, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
