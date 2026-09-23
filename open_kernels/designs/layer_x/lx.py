r"""lx: a whole linear-attention layer (attention block + MoE block) in ONE xclbin
context (phase 2 "whole-layer context", .claude/plans/open-kernels-phase2-whole-layer.md):

    ln -> gemv qkv | z -> glue -> [DeltaNet: its own context, for now] -> post -> gemv out
       -> ln (+residual) -> router -> MoE (8 routed experts + shared + combine)

The 8 main cores (one per column, Tile(c, mrow)) run every GEMV and the MoE in
one core program fed by three streams each: w (10 KB elements from the shim:
weights, the MoE header, experts), x (4 KB elements broadcast from the shim:
xn, og, xm, the expert hidden h) and y (256 B elements to the shim: band
results, the hidden parts, the block output). Helper cores: ln + router
(Tile(0, hrow)), post (Tile(1, hrow)), glue (Tile(2, hrow)). Shim budget: 13 fills,
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
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (Acquire, Bd, Buffer, CompileTime, DmaChannel, In, InOut, Lock, ObjectFifo, Out,
                     PacketFlow, Program, Release, Runtime, TaskGroup, TileDma, Worker)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_O, A_OG, A_OUT, A_QKV, A_RES, A_ROUT, A_VEC, A_XM, A_XN, A_Z, A_HP,  # noqa: E402
                    A_H, A_OUT2, C_BYTES, C_LNW, C_NW, C_POSTLN, C_RW, C_SGW, C_SIDE, C_WOUT,
                    ELN, GLUE_SIDE_BYTES, POOL_BYTES, POOL_FFN_DOWN, POOL_FFN_GATE, POOL_FFN_UP,
                    POOL_QKV, POOL_Z, SIDE_ALPHA, SIDE_BETA, SIDE_CONV, SIDE_SMALL,
                    STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES, R, SPEC)
import xcommon as X  # noqa: E402

D = R.linear
if D is None:
    sys.exit("lx.py: the spec has no linear-attention layers")
HID = X.HID
N_CORES = X.N_CORES
ELEM = X.ELEM
QKV_PC, Z_PC, OUT_PC = D.QKV_PC, D.Z_PC, D.OUT_PC      # bands per core: qkv (K = HID), z (K = HID), out (K = VW)
VW, OUT_K = D.VW, D.OUT_K
# dn_glue / dn_post
NCH, NHEAD = D.NCH, D.NHEAD
TILE, NT = D.TILE, D.NT
AB_ELEMS = D.AB_ELEMS
G, NG = D.G, D.NG
CONV_ROWS = SPEC.conv_kernel - 1                        # conv state rows (the taps before the new one)
KEY_TILES = D.VALUE_TILE0                               # tiles of the two key groups; the value tiles follow
VALUE_TILES = NT - KEY_TILES                            # tiles of the value group: NHEAD * value_dim / TILE.
                                                        # Equal to KEY_TILES only while NHEAD is 32 -- a
                                                        # 16-head model has 2 of them against 4 key tiles, and
                                                        # looping KEY_TILES twice made the core emit 32 records
                                                        # where the host drains 16 (the 2B / 0.8B hang,
                                                        # .claude/plans/q-hw-results.md section 3).
CONVW_ELEMS = SPEC.conv_kernel * TILE * 2 // ELEM       # 4 KB side elements holding one tile's conv taps
GLUE_NHEAD_DEFAULT = 32                                 # dn_glue.h's #ifndef DNGLUE_NHEAD value
DENSE = X.KIND == "dense"                     # the Qwen3.5 composition: a dense FFN tail, ONE stream
PART = int(os.environ.get("LX_PART", 0))
if X.ONDV:
    PART = 0          # the fused path ignores the part split: ONE stream runs the whole layer
STOP = int(os.environ.get("LX_STOP", 99))     # debug: truncate part 0 after the glue (1) / DeltaNet (2)
if DENSE and PART:
    sys.exit("lx.py: the dense tail is one instruction stream; LX_PART must be 0")
XN_ELEMS = D.XN_SIDE_ELEMS                    # 4 KB x / side elements the xn arrives in
OG_ELEMS = D.OG_ELEMS
# The alpha / beta weight tiles that belong to each 4 KB half of the xn (DENSE only: the glue
# core holds ONE 4 KB half at a time, so the projection is walked half by half). A tile is 64
# rows, a half carries up to 2048 of them, and at HID 2560 the two halves are 32 and 8.
AB_TILES = [min(ELEM // 2, HID - h * (ELEM // 2)) // 64 for h in range(XN_ELEMS)]
assert sum(AB_TILES) == AB_ELEMS, (AB_TILES, AB_ELEMS)
# dn_glue's head count. Passed ONLY when it differs from the header default, so the shipped
# 27B's five glue TUs keep the compile command they were built with (the DNX_PAD lesson).
GLUE_FLAGS = {} if NHEAD == GLUE_NHEAD_DEFAULT else {"compile_flags": [f"-DDNGLUE_NHEAD={NHEAD}"]}


def rows3(t: int):
    """Tile t (1024 bf16 = 2048 B) of each of the conv-state rows, in BYTES of the state BO
    (the conv state is its first STATE_S_OFF bytes; S follows)."""
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, STATE_BYTES), t * TILE * 2, [1, 1, CONV_ROWS, TILE * 2], [0, 0, NCH * 2, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"] + (["--generate-ctrl-pkt-overlay"] if os.environ.get("ONDV_CTRL_OVERLAY") == "1" else []))
def lx(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, *, part: CompileTime[int] = 0,
       stop: CompileTime[int] = 99, srchash: CompileTime[int] = 0):
    # the shipped design: the routed experts' fills are enqueued by the host and
    # moeroute2 repoints them between the two dispatches (see lx_ondv for the fused path)
    return _lx_build(pool, xres, consts, state, act, None, None, part=part, stop=stop, srchash=srchash, ondv=False, mrow=2, hrow=3)


def _lx_build(pool, xres, consts, state, act, cfg, octrl, *, part=0, stop=99, srchash=0, ondv=False, mrow=2, hrow=3, pieces=False):
    t = X.types()
    tl = X.ln_types()
    u8_4k = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    u8_2k = np.ndarray[(2048,), np.dtype[np.uint8]]
    u8_ln = tl["u8_ln"] if DENSE else u8_4k          # the norm helper's element (ELN bytes)
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(C_BYTES,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_BYTES,), np.dtype[np.uint8]]      # [conv state | S (in place)]
    act_ty = np.ndarray[(A_BYTES,), np.dtype[np.uint8]]
    nw_ty = np.ndarray[(SPEC.lin_value_dim,), np.dtype[bfloat16]]
    f32 = np.ndarray[(NHEAD,), np.dtype[np.float32]]
    fqk = np.ndarray[(2 * D.KEY_WIDTH,), np.dtype[np.float32]]
    fvt = np.ndarray[(TILE,), np.dtype[np.float32]]
    # The glue core's private copy of the layer-entry norm output. On the dense path it is
    # ONE 4 KB element (the projection is re-streamed per half): bf16[HID] costs the core
    # 8 192 B at HID 4096, 2 560 B more than it has.
    fxn = np.ndarray[(ELEM // 2 if DENSE else HID,), np.dtype[bfloat16]]

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(X.LN), str(X.RT), str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    if ondv:
        # the per-column control-word kernel (designs/router/ondv_ctrl_col.cc): its first two
        # args are the x broadcast's elements (bf16[2048] = 4 KB) and its fourth is the 512-B
        # control buffer (120 words + slack)
        f_oc = ExternalFunction("ondv_ctrl_col", source_file=str(X.RT / "ondv_ctrl_col.cc"),
                                arg_types=[t["x"], t["x"], np.int32, tl["u8_ctrl"]],
                                include_dirs=inc + [str(X.RT)],
                                compile_flags=[f"-DONDV_BD_UP={X.ONDV_BD_UP}",
                                               f"-DONDV_BD_GATE={X.ONDV_BD_GATE}",
                                               f"-DONDV_BD_DOWN={X.ONDV_BD_DOWN}"])
    f_ab = (ExternalFunction("glue_ab_e", source_file=str(GLUE / "glue_ab_e.cc"),
                             arg_types=[u8_4k, fxn, f32, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS) if DENSE else
            ExternalFunction("glue_ab", source_file=str(GLUE / "glue_ab.cc"), arg_types=[u8_4k, fxn, f32, np.int32], include_dirs=inc, **GLUE_FLAGS))
    f_small = ExternalFunction("glue_small_fn", source_file=str(GLUE / "glue_small.cc"), arg_types=[u8_4k, f32, f32, f32, f32], include_dirs=inc, **GLUE_FLAGS)
    f_conv = ExternalFunction("glue_conv", source_file=str(GLUE / "glue_conv.cc"),
                              arg_types=[u8_2k, u8_2k, u8_2k, u8_2k, u8_2k, u8_4k, u8_4k, u8_2k, u8_2k, u8_2k, fqk, fvt, np.int32, np.int32],
                              include_dirs=inc, **GLUE_FLAGS)
    f_emit = ExternalFunction("glue_emit_fn", source_file=str(GLUE / "glue_emit.cc"), arg_types=[fqk, fvt, f32, f32, u8_2k, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_copy = (ExternalFunction("glue_copy_xn_e", source_file=str(GLUE / "glue_copy_e.cc"),
                               arg_types=[u8_4k, fxn, np.int32], include_dirs=inc, **GLUE_FLAGS) if DENSE else
              ExternalFunction("glue_copy_xn", source_file=str(GLUE / "glue_copy.cc"),
                               arg_types=[u8_4k, fxn], include_dirs=inc, **GLUE_FLAGS))
    post_fn = ExternalFunction("post_fn", source_file=str(POST / "post.cc"), arg_types=[u8_4k, u8_4k, nw_ty, u8_2k], include_dirs=inc)
    post_copy = ExternalFunction("post_copy_nw", source_file=str(POST / "post_copy.cc"), arg_types=[u8_4k, nw_ty], include_dirs=inc)

    # ---- fifos
    # depth >= a routed band (8 elements) so a pinned routed descriptor, which transfers
    # a whole 81920-B band in ONE BD, fits the fifo the way the one-emitter probe's
    # single-element descriptor fits its depth-1 fifo (ONDV_W_DEPTH overrides)
    of_w = [ObjectFifo(t["elem"], name=f"w{c}", depth=int(os.environ.get("ONDV_W_DEPTH", 2)))
            for c in range(N_CORES)]
    of_y = [ObjectFifo(t["y"], name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(t["x"], name="x", depth=2)           # broadcast; og is acquired as 2 elements
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)        # [x0 x1 w] | [x0 x1 w a0 a1] | W x256
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1 if DENSE else 3)   # dense: one output element per call
    of_side = ObjectFifo(u8_4k, name="side", depth=2)
    of_gact = ObjectFifo(u8_2k, name="gact", depth=5)
    of_gout = ObjectFifo(u8_2k, name="gout", depth=3)
    of_pin = ObjectFifo(u8_4k, name="pin", depth=2)        # [nw][o g][z g]...
    of_pout = ObjectFifo(u8_2k, name="pout", depth=2)      # og per group
    # on-device routing (lx_ondv): eight per-column emitter cores at row 4, each running
    # ondv_ctrl_col and sending ONE packet-stamped BD on its own MM2S to its own column's
    # shim TileControl. ONE shared CoreTile object per column so the Worker, the Buffer, the
    # Lock, the TileDma and the PacketFlow all land on the SAME logical tile (the placer
    # dedups channel requirements by logical-tile op).
    emitter_tile = [Tile(c, int(os.environ.get("ONDV_EMITTER_ROW", 4)), tile_type=AIETileType.CoreTile)
                    for c in range(N_CORES)] if ondv else None
    # the shim tile each column's w fifo lives on. The packet flow's destination MUST be this
    # SAME Tile object (not a fresh Tile(c, 0)): the routed descriptors (BD 8/9/10) are written
    # on this shim's w channel, so the control packets must reach THIS shim's TileControl.
    shim_w = [Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(N_CORES)]
    ctrlw = [Buffer(tl["u8_ctrl"], name=f"ctrlw{c}", tile=emitter_tile[c]) for c in range(N_CORES)] if ondv else None
    # per-column acquire locks: index 0 fires slot 0's up|gate (right after rout+cfg); index
    # k+1 fires slot k's down AND slot k+1's up|gate on the h_k trigger (value 2 for k<7),
    # except index NE which fires only slot NE-1's down (value 1). One shared done lock per
    # column satisfies the verifier's acquire+release pairing on every packet BD.
    pktlk = [[Lock(emitter_tile[c], init=0, name=f"pktlk{c}_{i}") for i in range(X.NE + 1)] for c in range(N_CORES)] if ondv else None
    pktdone = [Lock(emitter_tile[c], init=0, name=f"pktdone{c}") for c in range(N_CORES)] if ondv else None

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

    def glue_body(sin, ain, oout, acc_a, acc_b, decay, beta, qk, vt, xn, fab, fsmall, fconv, femit, fcopy):
        if DENSE:
            # One accumulator at a time, one 4 KB half of the xn at a time: copy the half in
            # (so the fifo element can be released -- release(n) frees the OLDEST n), then run
            # that half's weight tiles off the same fifo. `first` resets the accumulator in the
            # first half only, so half 1 accumulates onto half 0's partial sum.
            for acc in (acc_a, acc_b):
                for h, ntiles in enumerate(AB_TILES):
                    e0 = sin.acquire(1)
                    fcopy(e0, xn, 0)
                    sin.release(1)
                    for tile in range_(ntiles):
                        ww = sin.acquire(1)
                        fab(ww, xn, acc, tile, 1 if h == 0 else 0)
                        sin.release(1)
        else:
            e0 = sin.acquire(1)
            fcopy(e0, xn)
            sin.release(1)
            for acc in (acc_a, acc_b):
                for tile in range_(AB_ELEMS):
                    ww = sin.acquire(1)
                    fab(ww, xn, acc, tile)
                    sin.release(1)
        sm = sin.acquire(1)
        fsmall(sm, acc_a, acc_b, decay, beta)
        sin.release(1)
        for base, ntiles in ((0, KEY_TILES), (KEY_TILES, VALUE_TILES)):
            for tt in range_(ntiles):
                ww = sin.acquire(CONVW_ELEMS)
                e = ain.acquire(2 + CONV_ROWS)
                o = oout.acquire(CONV_ROWS)
                fconv(e[0], e[1], e[2], e[3], e[4], ww[0], ww[1], o[0], o[1], o[2], qk, vt, tt, base)
                oout.release(CONV_ROWS)
                ain.release(2 + CONV_ROWS)
                sin.release(CONVW_ELEMS)
                if base == KEY_TILES:
                    for i in range_(D.HEADS_PER_TILE):
                        r = oout.acquire(1)
                        femit(qk, vt, decay, beta, r, tt, i)
                        oout.release(1)

    def post_body(ain, aout, nwb, f, fc):
        e = ain.acquire(1)
        fc(e, nwb)
        ain.release(1)
        for _ in range_(NG):
            e = ain.acquire(2)
            r = aout.acquire(1)
            f(e[0], e[1], nwb, r)
            aout.release(1)
            ain.release(2)

    workers = [Worker(X.ln_body, fn_args=[of_lni.cons(), of_lno.prod(), L["ln_nr"], L["ln_y"], L["ln_xn"]],
                      tile=Tile(0, hrow), stack_size=0x1800)
               if DENSE else
               Worker(X.ln_router_body,
                      fn_args=[of_lni.cons(), of_lno.prod(), Buffer(tl["xb"], name="rxs"), Buffer(tl["racc"], name="racc"),
                               L["ln_nr"], L["ln"], L["rcopy"], L["racc"], L["rfin"]],
                      tile=Tile(0, hrow), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body,
                              fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), *X.worker_args(X.core_buffers(t, c), K)],
                              tile=Tile(c, mrow), stack_size=0x1800))
    workers.append(Worker(post_body, fn_args=[of_pin.cons(), of_pout.prod(), Buffer(nw_ty, name="nwb"), post_fn, post_copy],
                          tile=Tile(1, hrow), stack_size=0x1800))
    workers.append(Worker(glue_body,
                          fn_args=[of_side.cons(), of_gact.cons(), of_gout.prod(),
                                   Buffer(f32, name="acc_a"), Buffer(f32, name="acc_b"), Buffer(f32, name="decay"),
                                   Buffer(f32, name="beta"), Buffer(fqk, name="qk"), Buffer(fvt, name="vt"), Buffer(fxn, name="xnb"),
                                   f_ab, f_small, f_conv, f_emit, f_copy],
                          tile=Tile(2, hrow), stack_size=0x1800))
    if ondv:
        # x-broadcast elements before the two on-device-routing elements (the router's top-8
        # and the 2-word pool base), which now arrive right after the MoE header: part 0's xn
        # + og, then moe_sequence's xm. The per-wave h elements follow, after rout + cfg.
        N_X_SKIP = XN_ELEMS + OG_ELEMS + 1

        def _emitter_body(c):
            def emitter_body(xin, ctrl, f_oc_col, *locks):
                for _ in range_(N_X_SKIP):
                    e = xin.acquire(1)
                    xin.release(1)
                r = xin.acquire(1)          # act[A_ROUT] (the router's top-8)
                cfg = xin.acquire(1)        # the 2-word pool base
                f_oc_col(r, cfg, c, ctrl)
                xin.release(2)
                # ONDV_DONE_ACQ: the working one-emitter probe (designs/expert_fetch/
                # ondv_live_probe.py) pairs every packet BD's release(pktdone) with an
                # acquire in the core. The shipped emitter only releases the shared done
                # lock, so the BD's release has no matching acquire.
                done = locks[X.NE + 1] if len(locks) > X.NE + 1 else None
                locks[0].release(1)         # slot 0's up|gate (BD 8/9)
                if done is not None:
                    done.acquire(1)
                for e in range(X.NE):       # Python-unrolled: h_0 .. h_7
                    h = xin.acquire(1)
                    xin.release(1)
                    # h_k fires slot k's down AND slot k+1's up|gate (one 15-word packet),
                    # except h_{NE-1} fires only the last down (a 5-word packet)
                    locks[e + 1].release(1)
                    if done is not None:
                        done.acquire(1)
            return emitter_body

        if os.environ.get("ONDV_NO_EMITTERS") != "1":
            for c in X.ONDV_EMITTER_COLS:
                eargs = [of_x.cons(), ctrlw[c], f_oc, *pktlk[c]]
                if os.environ.get("ONDV_DONE_ACQ") == "1":
                    eargs.append(pktdone[c])
                workers.append(Worker(_emitter_body(c), fn_args=eargs,
                                      tile=emitter_tile[c], stack_size=0x1800))

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

    def sequence(*a):
        if ondv:
            (a_pool, c_xres, a_consts, a_state, a_act, a_cfg) = a[:6]
            (lni, lno, w_prods, x_prod, y_conss, side_p, gact_p, gout_c, pin_p, pout_c) = a[6:16]
        else:
            (a_pool, c_xres, a_consts, a_state, a_act) = a[:5]
            (lni, lno, w_prods, x_prod, y_conss, side_p, gact_p, gout_c, pin_p, pout_c) = a[5:15]
            a_cfg = None
        if DENSE:
            dense_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                           side_p, gact_p, gout_c, pin_p, pout_c)
        elif part == 0 or ondv:
            # With MOE_ONDEVICE_ROUTE this is the WHOLE layer in ONE instruction stream:
            # the per-column emitter cores retarget + push the routed-expert descriptors
            # on-device, so there is no host between the router and the routed experts and
            # no second dispatch. Without it, `part` still selects the half of the layer the
            # driver dispatches and moeroute2 patches between.
            # 1. layer-entry norm: xn -> act[A_XN]
            tg_ln = TaskGroup()
            lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
            lni.fill(a_consts, tap=bt(C_BYTES, C_LNW, ELEM), wait=True, group=tg_ln)
            lno.drain(a_act, tap=bt(A_BYTES, A_XN, ELEM), wait=True, group=tg_ln)
            # 2. qkv | z GEMV: weights now, x after the norm
            pw, py = Pipeline(3), Pipeline(3)
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
                py.drain(y_conss[c], a_act, bt(A_BYTES, A_QKV + c * QKV_PC * YB, QKV_PC * YB))
                py.drain(y_conss[c], a_act, bt(A_BYTES, A_Z + c * Z_PC * YB, Z_PC * YB))
            tg_ln.finish()                                   # xn is in DDR
            px = Pipeline(3)
            px.fill(x_prod, a_act, bt(A_BYTES, A_XN, ELEM))
            tg_s = TaskGroup()
            side_p.fill(a_act, tap=bt(A_BYTES, A_XN, ELEM), wait=True, group=tg_s)
            side_p.fill(a_consts, tap=bt(C_BYTES, C_SIDE, GLUE_SIDE_BYTES), wait=True, group=tg_s)
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
            tg_s.finish()
            if STOP == 1:
                pw.finish()
                px.finish()
                return
            # 4. DeltaNet on the main cores: S in place, o -> act[A_O]
            X.dn_sequence(pw, py, a_state, a_act, w_prods, y_conss, A_BYTES, A_VEC, A_O, STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES)
            py.finish()                                      # o is in DDR
            if STOP == 2:
                pw.finish()
                px.finish()
                return
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
            px.fill(x_prod, a_act, bt(A_BYTES, A_OG, VW * 2))
            py.finish()                                      # out is in DDR
            # 7. residual + post-attention norm, then the router
            tg_ln = TaskGroup()
            lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
            lni.fill(a_consts, tap=bt(C_BYTES, C_POSTLN, ELEM), wait=True, group=tg_ln)
            lni.fill(a_act, tap=bt(A_BYTES, A_OUT, HID * 4), wait=True, group=tg_ln)
            lno.drain(a_act, tap=bt(A_BYTES, A_RES, HID * 4), wait=True, group=tg_ln)
            lno.drain(a_act, tap=bt(A_BYTES, A_XM, ELEM), wait=True, group=tg_ln)
            tg_ln.finish()
            tg_r = TaskGroup()
            lni.fill(a_consts, tap=bt(C_BYTES, C_RW, X.W_ELEMS * ELEM), wait=True, group=tg_r)
            lno.drain(a_act, tap=bt(A_BYTES, A_ROUT, ELEM), wait=True, group=tg_r)
            tg_r.finish()
            pw.finish()
            px.finish()
            if ondv and os.environ.get("ONDV_SKIP_MOE") != "1":
                # the rest of the layer, one stream: the routed fills are configured but
                # never enqueued and the per-column emitter cores retarget + push them
                # on-device (their rout + pool-base input rides the x broadcast)
                X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                               A_BYTES, C_BYTES, A_XM, A_ROUT, A_RES, A_HP, C_SGW,
                               ondv=(a_cfg,))
        else:
            # 8. the MoE block (moeroute2 has pointed the routed slots' fills at the router's choice)
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                           A_BYTES, C_BYTES, A_XM, A_ROUT, A_RES, A_HP, C_SGW)

    rt_args = [pool_ty, xres_ty, consts_ty, state_ty, act_ty]
    if ondv:
        # one extra DDR buffer: the pool-base config in (the 2 words the emitters retarget
        # the routed-expert descriptors against; the router's top-8 rides `act` at A_ROUT)
        rt_args += [np.ndarray[(ELEM,), np.dtype[np.uint8]]]
    rt_args += [of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                [of_w[c].prod(tile=shim_w[c] if ondv else Tile(c, 0)) for c in range(N_CORES)],
                of_x.prod(tile=Tile(1, 0)),
                [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)), of_gout.cons(tile=Tile(2, 0)),
                of_pin.prod(tile=Tile(4, 0)), of_pout.cons(tile=Tile(1, 0))]
    rt = Runtime(sequence, rt_args)
    flows = []
    if ondv:
        # one packet-stamped BD per emitter core on its OWN MM2S ch1, aimed at its own
        # column's shim TileControl (pkt_id 15 = the placer's controller_id). A core's
        # packet reaches its own column on South -- the legal in-column arrival -- so no
        # cross-column routing, no control overlay and no shim MM2S channel are needed.
        for c in X.ONDV_EMITTER_COLS:
            for lk in pktlk[c]:
                rt.add_lock(lk)
            rt.add_lock(pktdone[c])
            # 9 packet BDs per emitter, chained in firing order and paced by the acquire
            # locks the emitter releases: slot 0's up|gate (56 B), then per slot k>=1 an
            # 84-B packet holding slot k-1's down (28 B) + slot k's up|gate (56 B), and
            # finally slot NE-1's down (28 B). Each reads its column's ctrlw slice at a
            # byte offset; the down_{k-1}|up|gate_k words are contiguous (offset 84k-28).
            # The BDs are NOT packet-stamped: the control stream carries its own stream
            # headers (ondv_ctrl.h) and the flow below keeps them.
            bds = [Bd(buffer=ctrlw[c], offset=0, length=56,
                      acquires=[Acquire(pktlk[c][0])], releases=[Release(pktdone[c])], next=1)]
            for k in range(1, X.NE):
                bds.append(Bd(buffer=ctrlw[c], offset=84 * k - 28, length=84,
                              acquires=[Acquire(pktlk[c][k])], releases=[Release(pktdone[c])],
                              next=k + 1))
            bds.append(Bd(buffer=ctrlw[c], offset=644, length=28,
                          acquires=[Acquire(pktlk[c][X.NE])], releases=[Release(pktdone[c])], next=0))
            rt.add_tile_dma(TileDma(
                tile=emitter_tile[c],
                channels=[DmaChannel(direction=DMAChannelDir.MM2S, channel=1, bds=bds)]))
            flows.append(PacketFlow(pkt_id=15, src=emitter_tile[c],
                                    src_port=WireBundle.DMA, src_channel=1,
                                    dst=shim_w[c],
                                    dst_port=WireBundle.TileControl, dst_channel=0,
                                    keep_pkt_header=True))
    for f in flows:
        rt.add_flow(f)
    if pieces:
        # the merged design (lax.py) needs the parts, not a Program: one xclbin must hold
        # both layer types, and only the instruction stream differs between them
        return workers, rt_args, flows, sequence
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"] + (["--generate-ctrl-pkt-overlay"] if os.environ.get("ONDV_CTRL_OVERLAY") == "1" else []))
def lx_ondv(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, cfg: In, *,
            part: CompileTime[int] = 0, stop: CompileTime[int] = 99, srchash: CompileTime[int] = 0):
    """The fused whole-layer path: eight per-column emitter cores write the routed-expert
    retarget+enqueue control words and send them on-device, so the layer is ONE instruction
    stream (no host between the router and the routed experts).

    One extra DDR buffer: `cfg` carries [base_lo, base_hi] = the MoE pool BO's DDR address
    (bo.address() + 0x8000_0000; the driver writes it, one per layer). The router's top-8
    rides `act` at A_ROUT and reaches the emitters over the x broadcast."""
    return _lx_build(pool, xres, consts, state, act, cfg, None, part=part, stop=stop, srchash=srchash, ondv=True, mrow=2, hrow=3)


DESIGN = lx_ondv if X.ONDV else lx
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes()] + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "stop": STOP, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
