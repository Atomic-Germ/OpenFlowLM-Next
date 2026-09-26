r"""lx: a whole linear-attention layer (attention block + MoE block) in ONE xclbin
context (phase 2 "whole-layer context", .claude/plans/open-kernels-phase2-whole-layer.md):

    ln -> gemv qkv | z -> glue -> [DeltaNet: its own context, for now] -> post -> gemv out
       -> ln (+residual) -> router -> MoE (8 routed experts + shared + combine)

The 8 main cores (one per column, Tile(c, 2)) run every GEMV and the MoE in
one core program fed by three streams each: w (10 KB elements from the shim:
weights, the MoE header, experts), x (4 KB elements broadcast from the shim:
xn, og, xm, the expert hidden h) and y (256 B elements to the shim: band
results, the hidden parts, the block output). Helper cores: ln + router
(Tile(0, 3)), post (Tile(1, 3)), glue (Tile(2, 3)). Shim budget: 13 fills,
11 drains. Cores do not know about dispatch boundaries -- they block on the
next element -- so one xclbin serves THREE instruction streams (CompileTime
`part`): 0 = ln -> qkv|z -> glue (the DeltaNet step runs in between, in
designs/deltanet's context, on `act`), 1 = post -> out -> ln -> router, 2 = the
MoE (the driver's `moeroute2` patches the routed slots' fills from the router
output between parts 1 and 2). Build all three; they share part 0's xclbin.

Geometry: the recipe's (open_kernels/recipes/qwen36moe.py) for the spec named
by OPEN_KERNELS_SPEC (else the checked-in 27B) -- layout.py for the byte
layouts, xcommon.py for the main-core streams, `R.linear` for this layer type.

Args (layout.py): pool u8[POOL_BYTES] (qkv, z, experts at their pool offsets),
xres f32[HID] (in: the layer input; out: the layer output), consts (per layer),
state (conv state + S, in place), act (scratch; vec/o for DeltaNet).
Build (WSL): for p in 0 1 2: LX_PART=$p python build_design.py designs/layer_x/lx.py designs/layer_x/build_lx$p
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, InOut, Program, Runtime, TaskGroup
from aie.iron.device import Tile

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_O, A_OG, A_OUT, A_QKV, A_RES, A_VEC, A_XM, A_XN, A_Z, A_HP,  # noqa: E402
                    A_H, A_OUT2, A_ROUT, C_BYTES, C_LNW, C_NW, C_POSTLN, C_SGW, C_SIDE, C_WOUT,
                    ELN, GLUE_SIDE_BYTES, POOL_BYTES, POOL_FFN_DOWN, POOL_FFN_GATE, POOL_FFN_UP,
                    POOL_QKV, POOL_Z, SIDE_ALPHA, SIDE_BETA, SIDE_CONV, SIDE_SMALL,
                    STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES, R, SPEC)
import xcommon as X  # noqa: E402
import xlayer as XL  # noqa: E402

D = R.linear
if D is None:
    sys.exit("lx.py: the spec has no linear-attention layers")
HID = X.HID
N_CORES = X.N_CORES
ELEM = X.ELEM
QKV_PC, Z_PC, OUT_PC = D.QKV_PC, D.Z_PC, D.OUT_PC      # bands per core: qkv (K = HID), z (K = HID), out (K = VW)
VW, OUT_K = D.VW, D.OUT_K
# dn_glue / dn_post: their constants, kernels, bodies and this layer's part-0 host sequence
# live in xlayer.py, shared with the merged image (ux.py)
TILE, NT, G, NG, KEY_TILES = XL.TILE, XL.NT, XL.G, XL.NG, XL.KEY_TILES
AB_TILES, rows3 = XL.AB_TILES, XL.rows3
DENSE = X.KIND == "dense"                     # the Qwen3.5 composition: a dense FFN tail, ONE stream
PART = int(os.environ.get("LX_PART", 0))
STOP = int(os.environ.get("LX_STOP", 99))     # debug: truncate part 0 after the glue (1) / DeltaNet (2)
if DENSE and PART:
    sys.exit("lx.py: the dense tail is one instruction stream; LX_PART must be 0")
XN_ELEMS = D.XN_SIDE_ELEMS                    # 4 KB x / side elements the xn arrives in
OG_ELEMS = D.OG_ELEMS


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lx(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, *, part: CompileTime[int] = 0,
       stop: CompileTime[int] = 99, srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    g = XL.glue_post_types()
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(C_BYTES,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_BYTES,), np.dtype[np.uint8]]      # [conv state | S (in place)]
    act_ty = np.ndarray[(A_BYTES,), np.dtype[np.uint8]]

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(X.LN), str(X.RT), str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    GK = XL.glue_post_kernels(inc, g)

    # ---- fifos
    of_w, of_y, of_x = XL.main_fifos(t)
    of_lni, of_lno = XL.ln_fifos(tl)
    of_side, of_gact, of_gout, of_pin, of_pout = XL.glue_post_fifos(g)

    # ---- cores
    def main_body(win, xin, yout, *args):
        B, K = X.unpack_args(args)
        tab = B["tab"]
        if DENSE:
            # ONE stream: qkv | z, DeltaNet, the out projection, then the dense FFN tail.
            X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS, QKV_PC + Z_PC, "linear")
            X.dn_body(win, yout, B, K)
            X.prep_bands(win, xin, yout, B, K, OUT_K, OG_ELEMS, OUT_PC, "linear_out")
            X.ffn_body(win, xin, yout, B, K)
            return
        # part 0: qkv | z against xn, then this core's DeltaNet heads
        xe = xin.acquire(1)
        K["prep2048"](xe, tab)
        X.role_gemv_bands(win, yout, B, K, "linear", QKV_PC + Z_PC, HID)
        xin.release(1)
        X.dn_body(win, yout, B, K)
        # (still part 0) out against og (two 4 KB elements, K = VW)
        oe = xin.acquire(2)
        K["prep4096a"](oe[0], tab)
        K["prep4096b"](oe[1], tab)
        X.role_gemv_bands(win, yout, B, K, "linear_out", OUT_PC, OUT_K)
        xin.release(2)
        # part 1: the MoE block
        X.moe_body(win, xin, yout, B, K)

    workers = ([XL.ln_worker(of_lni, of_lno, tl, L)] + XL.main_workers(main_body, of_w, of_x, of_y, t, K)
               + XL.glue_post_workers(of_side, of_gact, of_gout, of_pin, of_pout, g, GK, Tile(2, 3)))

    bt = X.bt
    BB_HID, BB_OUT = X.role_band_bytes("linear", HID), X.role_band_bytes("linear_out", OUT_K)
    YB = X.BAND_ROWS * 4                                   # one band's y bytes

    # ---- host sequences (one per instruction stream)
    def dense_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                       side_p, gact_p, gout_c, pin_p, pout_c):
        """ONE instruction stream: the MoE stream's steps 1-6 with the router dropped, then
        designs/dense/dx.py's steps 5-7 (residual + norm, the FFN, the output residual)."""
        # 1. layer-entry norm: xn -> act[A_XN]
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(C_BYTES, C_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(A_BYTES, A_XN, ELN), wait=True, group=tg_ln)
        # 2. qkv | z GEMV: weights now, x after the norm
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
            py.drain(y_conss[c], a_act, bt(A_BYTES, A_QKV + c * QKV_PC * YB, QKV_PC * YB))
            py.drain(y_conss[c], a_act, bt(A_BYTES, A_Z + c * Z_PC * YB, Z_PC * YB))
        tg_ln.finish()                                   # xn is in DDR
        px.fill(x_prod, a_act, bt(A_BYTES, A_XN, XN_ELEMS * ELEM))
        # The glue's side channel, in the order the core acquires it: per accumulator, each
        # 4 KB half of the xn then that half's weight tiles, and last `small` and the conv
        # taps. Throttled like every other channel -- a shim channel's start queue holds 4 BDs
        # and one TaskGroup of 10 would silently drop the rest (ironutil.Pipeline). The count
        # is `qwen35.glue_side_fills`, checked against the shim budget by the recipe.
        ps = Pipeline(3)
        for reg in (SIDE_ALPHA, SIDE_BETA):
            off = 0
            for h, ntiles in enumerate(AB_TILES):
                ps.fill(side_p, a_act, bt(A_BYTES, A_XN + h * ELEM, ELEM))
                ps.fill(side_p, a_consts, bt(C_BYTES, C_SIDE + reg + off, ntiles * ELEM))
                off += ntiles * ELEM
        ps.fill(side_p, a_consts, bt(C_BYTES, C_SIDE + SIDE_SMALL, ELEM))
        ps.fill(side_p, a_consts, bt(C_BYTES, C_SIDE + SIDE_CONV, GLUE_SIDE_BYTES - SIDE_CONV))
        py.finish()                                      # qkv, z are in DDR
        # 3. glue: conv state updated in place, DeltaNet records -> act[A_VEC]
        pipe = Pipeline(3)
        for tt in range(NT):
            pipe.drain(gout_c, a_state, rows3(tt))
            if tt >= KEY_TILES:
                pipe.drain(gout_c, a_act, bt(A_BYTES, A_VEC + (tt - KEY_TILES) * D.HEADS_PER_TILE * D.RECORD_BYTES,
                                             D.HEADS_PER_TILE * D.RECORD_BYTES))
            pipe.fill(gact_p, a_act, bt(A_BYTES, A_QKV + tt * TILE * 4, TILE * 4))
            pipe.fill(gact_p, a_state, rows3(tt))
        pipe.finish()                                    # the records are in DDR
        ps.finish()
        # 4. DeltaNet on the main cores: S in place, o -> act[A_O]
        X.dn_sequence(pw, py, a_state, a_act, w_prods, y_conss, A_BYTES, A_VEC, A_O, STATE_BYTES, STATE_S_OFF,
                      S_HEAD_BYTES)
        py.finish()                                      # o is in DDR
        # 5. post: og -> act[A_OG] (z from act, o from DeltaNet)
        pipe = Pipeline(3)
        pipe.fill(pin_p, a_consts, bt(C_BYTES, C_NW, ELEM))
        for g in range(NG):
            pipe.drain(pout_c, a_act, bt(A_BYTES, A_OG + g * G * 2, G * 2))
            pipe.fill(pin_p, a_act, bt(A_BYTES, A_O + g * G * 4, G * 4))
            pipe.fill(pin_p, a_act, bt(A_BYTES, A_Z + g * G * 4, G * 4))
        pipe.finish()                                    # og is in DDR
        # 6. out projection (weights in consts) against og
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_consts, bt(C_BYTES, C_WOUT + c * OUT_PC * BB_OUT, OUT_PC * BB_OUT))
            py.drain(y_conss[c], a_act, bt(A_BYTES, A_OUT + c * OUT_PC * YB, OUT_PC * YB))
        px.fill(x_prod, a_act, bt(A_BYTES, A_OG, OG_ELEMS * ELEM))
        py.finish()                                      # out is in DDR
        # 7. res = xres + out; xm = post_attention_norm(res)  (three output elements, one per call)
        tg_ln2 = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(C_BYTES, C_POSTLN, ELN), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(A_BYTES, A_RES, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(A_BYTES, A_XM, ELN), wait=True, group=tg_ln2)
        lni.fill(a_act, tap=bt(A_BYTES, A_OUT, HID * 4), wait=True, group=tg_ln2)
        tg_ln2.finish()                                  # res, xm are in DDR
        # 8. the dense FFN: up | gate -> h, down -> out2
        X.ffn_sequence(pw, px, py, a_pool, a_act, w_prods, x_prod, y_conss,
                       A_BYTES, A_XM, A_H, A_OUT2, POOL_FFN_UP, POOL_FFN_GATE, POOL_FFN_DOWN)
        # 9. xres = res + out2 (the norm output is junk; nothing reads it)
        tg_ln3 = TaskGroup()
        lni.fill(a_act, tap=bt(A_BYTES, A_RES, HID * 4), wait=True, group=tg_ln3)
        lni.fill(a_consts, tap=bt(C_BYTES, C_POSTLN, ELN), wait=True, group=tg_ln3)   # unused w
        lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln3)
        lno.drain(a_act, tap=bt(A_BYTES, A_XN, ELN), wait=True, group=tg_ln3)   # the junk xn, over the spent A_XN
        py.finish()                                      # out2 is in DDR
        lni.fill(a_act, tap=bt(A_BYTES, A_OUT2, HID * 4), wait=True, group=tg_ln3)
        tg_ln3.finish()
        pw.finish()
        px.finish()

    def sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss, side_p, gact_p, gout_c, pin_p, pout_c):
        if DENSE:
            dense_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                           side_p, gact_p, gout_c, pin_p, pout_c)
        elif part == 0:
            XL.linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                               side_p, gact_p, gout_c, pin_p, pout_c, stop=STOP)
        else:
            # 8. the MoE block (moeroute2 has pointed the routed slots' fills at the router's choice)
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                           A_BYTES, C_BYTES, A_XM, A_ROUT, A_RES, A_HP, C_SGW)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)), of_gout.cons(tile=Tile(2, 0)),
                            of_pin.prod(tile=Tile(4, 0)), of_pout.cons(tile=Tile(1, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = lx
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes(), (HERE / "xlayer.py").read_bytes()] + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "stop": STOP, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
