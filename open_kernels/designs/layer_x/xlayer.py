r"""The per-layer-type pieces the whole-layer designs share: the helper cores of the
linear-attention layer (glue, post) and of the full-attention layer (the attention
cores), the norm helper, the main-core and helper fifos, and the part-0 host
sequence of each layer type. xcommon.py holds what every stream of every design
shares (the main-core kernels, the MoE and DeltaNet fragments); this module holds
what used to be written out twice because it lived inside a design's `@iron.jit`
function:

    lx.py   linear layer,      two contexts   glue + post + linear_sequence
    ax.py   full layer,        two contexts   attention + full_sequence
    ux.py   both on one image (the merged decode layer): all of the above

ux.py used to COPY lx.py's glue / post bodies and host sequence and ax.py's attention
bodies and host sequence. An edit to one design that was not hand-mirrored into ux.py
shipped stale code in the merged image (it happened twice in the decode-gap work).
Now there is one copy, here, and the three designs call it.

Every function emits exactly what the design emitted when the code was inline -- the
same ops, the same objects created in the same order, the same names. That is checked
by building every family's kernels before and after and comparing the bytes
(export_qwen36_kernels.py --check); anything added here has to keep that true for
every design that calls it, not only the one being changed.

The buffer totals (`act_t`, `consts_t`, `state_t`) are parameters because the merged
image declares ONE set of buffer-argument types for both layer types, the larger of
each pair; a tap's `total` only bounds its offsets, so the lx / ax values and the
merged values describe the same transfers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, ObjectFifo, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction

from ironutil import Pipeline
from layout import (A_BYTES, A_O, A_OG, A_OUT, A_QKV, A_RES, A_ROUT, A_VEC, A_XM, A_XN, A_Z,  # noqa: E402
                    AA_BYTES, AA_KVN, AA_OG, AA_OUT, AA_QG, AA_RES, AA_ROUT, AA_XM, AA_XN,
                    C_BYTES, C_LNW, C_NW, C_POSTLN, C_RW, C_SIDE, C_WOUT,
                    CA_BYTES, CA_LNW, CA_META, CA_POSTLN, CA_RW, GLUE_SIDE_BYTES, KV_BYTES, KV_ROW,
                    POOL_BYTES, POOL_GATE, POOL_K, POOL_O, POOL_Q, POOL_QKV, POOL_V, POOL_Z,
                    PTAB_BYTES, PTAB_ROW, STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES, R, SPEC)
import xcommon as X  # noqa: E402

HERE = Path(__file__).parent
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
ATTN = HERE.parent / "attn"

HID, ELEM, N_CORES = X.HID, X.ELEM, X.N_CORES
DENSE = X.KIND == "dense"                     # the Qwen3.5 composition: a dense FFN tail
STACK = 0x1800
bt = X.bt
YB = X.BAND_ROWS * 4                          # one band's y bytes


# ---- the main cores' streams and the norm helper (every layer type)
def main_fifos(t):
    of_w = [ObjectFifo(t["elem"], name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(t["y"], name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(t["x"], name="x", depth=2)           # broadcast; og is acquired as 2 elements
    return of_w, of_y, of_x


def ln_fifos(tl):
    u8_ln = tl["u8_ln"] if DENSE else np.ndarray[(ELEM,), np.dtype[np.uint8]]   # the norm helper's element
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)        # [x0 x1 w] | [x0 x1 w a0 a1] | W x256
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1 if DENSE else 3)   # dense: one output element per call
    return of_lni, of_lno


def ln_worker(of_lni, of_lno, tl, L):
    """The norm (+ router) helper at Tile(0, 3)."""
    if DENSE:
        return Worker(X.ln_body, fn_args=[of_lni.cons(), of_lno.prod(), L["ln_nr"], L["ln_y"], L["ln_xn"]],
                      tile=Tile(0, 3), stack_size=STACK)
    return Worker(X.ln_router_body,
                  fn_args=[of_lni.cons(), of_lno.prod(), Buffer(tl["xb"], name="rxs"), Buffer(tl["racc"], name="racc"),
                           L["ln_nr"], L["ln"], L["rcopy"], L["racc"], L["rfin"]],
                  tile=Tile(0, 3), stack_size=STACK)


def main_workers(body, of_w, of_x, of_y, t, K, extra=lambda c: []):
    """The eight main cores, Tile(c, 2). `extra(c)` goes between the streams and the
    buffers (the merged image's RTP words)."""
    return [Worker(body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), *extra(c),
                                  *X.worker_args(X.core_buffers(t, c), K)],
                   tile=Tile(c, 2), stack_size=STACK)
            for c in range(N_CORES)]


# ---- the linear-attention layer's helpers: dn_glue (Tile(2, 3) in lx) and dn_post (Tile(1, 3))
D = R.linear
if D is not None:
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
    XN_ELEMS = D.XN_SIDE_ELEMS                              # 4 KB x / side elements the xn arrives in
    # The alpha / beta weight tiles that belong to each 4 KB half of the xn (DENSE only: the glue
    # core holds ONE 4 KB half at a time, so the projection is walked half by half). A tile is 64
    # rows, a half carries up to 2048 of them, and at HID 2560 the two halves are 32 and 8.
    AB_TILES = [min(ELEM // 2, HID - h * (ELEM // 2)) // 64 for h in range(XN_ELEMS)]
    assert sum(AB_TILES) == AB_ELEMS, (AB_TILES, AB_ELEMS)
    # dn_glue's head count. Passed ONLY when it differs from the header default, so the shipped
    # 27B's five glue TUs keep the compile command they were built with (the DNX_PAD lesson).
    GLUE_FLAGS = {} if NHEAD == GLUE_NHEAD_DEFAULT else {"compile_flags": [f"-DDNGLUE_NHEAD={NHEAD}"]}


def rows3(t: int, state_t: int = STATE_BYTES):
    """Tile t (1024 bf16 = 2048 B) of each of the conv-state rows, in BYTES of the state BO
    (the conv state is its first STATE_S_OFF bytes; S follows). `state_t` is the declared
    state type's size: STATE_BYTES in lx, the larger of it and KV_BYTES in the merged image."""
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, state_t), t * TILE * 2, [1, 1, CONV_ROWS, TILE * 2], [0, 0, NCH * 2, 1])


def glue_post_types():
    g = {}
    g["u8_4k"] = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    g["u8_2k"] = np.ndarray[(2048,), np.dtype[np.uint8]]
    g["nw"] = np.ndarray[(SPEC.lin_value_dim,), np.dtype[bfloat16]]
    g["f32"] = np.ndarray[(NHEAD,), np.dtype[np.float32]]
    g["fqk"] = np.ndarray[(2 * D.KEY_WIDTH,), np.dtype[np.float32]]
    g["fvt"] = np.ndarray[(TILE,), np.dtype[np.float32]]
    # The glue core's private copy of the layer-entry norm output. On the dense path it is
    # ONE 4 KB element (the projection is re-streamed per half): bf16[HID] costs the core
    # 8 192 B at HID 4096, 2 560 B more than it has.
    g["fxn"] = np.ndarray[(ELEM // 2 if DENSE else HID,), np.dtype[bfloat16]]
    return g


def glue_post_kernels(inc, g):
    u8_4k, u8_2k, f32, fqk, fvt, fxn, nw_ty = g["u8_4k"], g["u8_2k"], g["f32"], g["fqk"], g["fvt"], g["fxn"], g["nw"]
    k = {}
    k["ab"] = (ExternalFunction("glue_ab_e", source_file=str(GLUE / "glue_ab_e.cc"),
                                arg_types=[u8_4k, fxn, f32, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
               if DENSE else
               ExternalFunction("glue_ab", source_file=str(GLUE / "glue_ab.cc"), arg_types=[u8_4k, fxn, f32, np.int32],
                                include_dirs=inc, **GLUE_FLAGS))
    k["small"] = ExternalFunction("glue_small_fn", source_file=str(GLUE / "glue_small.cc"),
                                  arg_types=[u8_4k, f32, f32, f32, f32], include_dirs=inc, **GLUE_FLAGS)
    k["conv"] = ExternalFunction("glue_conv", source_file=str(GLUE / "glue_conv.cc"),
                                 arg_types=[u8_2k, u8_2k, u8_2k, u8_2k, u8_2k, u8_4k, u8_4k, u8_2k, u8_2k, u8_2k,
                                            fqk, fvt, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    k["emit"] = ExternalFunction("glue_emit_fn", source_file=str(GLUE / "glue_emit.cc"),
                                 arg_types=[fqk, fvt, f32, f32, u8_2k, np.int32, np.int32], include_dirs=inc,
                                 **GLUE_FLAGS)
    k["copy"] = (ExternalFunction("glue_copy_xn_e", source_file=str(GLUE / "glue_copy_e.cc"),
                                  arg_types=[u8_4k, fxn, np.int32], include_dirs=inc, **GLUE_FLAGS) if DENSE else
                 ExternalFunction("glue_copy_xn", source_file=str(GLUE / "glue_copy.cc"),
                                  arg_types=[u8_4k, fxn], include_dirs=inc, **GLUE_FLAGS))
    k["post"] = ExternalFunction("post_fn", source_file=str(POST / "post.cc"), arg_types=[u8_4k, u8_4k, nw_ty, u8_2k],
                                 include_dirs=inc)
    k["post_copy"] = ExternalFunction("post_copy_nw", source_file=str(POST / "post_copy.cc"), arg_types=[u8_4k, nw_ty],
                                      include_dirs=inc)
    return k


def glue_post_fifos(g):
    of_side = ObjectFifo(g["u8_4k"], name="side", depth=2)
    of_gact = ObjectFifo(g["u8_2k"], name="gact", depth=5)
    of_gout = ObjectFifo(g["u8_2k"], name="gout", depth=3)
    of_pin = ObjectFifo(g["u8_4k"], name="pin", depth=2)        # [nw][o g][z g]...
    of_pout = ObjectFifo(g["u8_2k"], name="pout", depth=2)      # og per group
    return of_side, of_gact, of_gout, of_pin, of_pout


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


def glue_post_workers(of_side, of_gact, of_gout, of_pin, of_pout, g, k, glue_tile):
    """[post at Tile(1, 3), glue at `glue_tile`] -- in that order. lx puts the glue at
    Tile(2, 3); the merged image moves it to Tile(2, 4), where it does not take the
    attention core's tile."""
    f32 = g["f32"]
    return [Worker(post_body, fn_args=[of_pin.cons(), of_pout.prod(), Buffer(g["nw"], name="nwb"),
                                       k["post"], k["post_copy"]],
                   tile=Tile(1, 3), stack_size=STACK),
            Worker(glue_body,
                   fn_args=[of_side.cons(), of_gact.cons(), of_gout.prod(),
                            Buffer(f32, name="acc_a"), Buffer(f32, name="acc_b"), Buffer(f32, name="decay"),
                            Buffer(f32, name="beta"), Buffer(g["fqk"], name="qk"), Buffer(g["fvt"], name="vt"),
                            Buffer(g["fxn"], name="xnb"),
                            k["ab"], k["small"], k["conv"], k["emit"], k["copy"]],
                   tile=glue_tile, stack_size=STACK)]


def linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                    side_p, gact_p, gout_c, pin_p, pout_c,
                    act_t=A_BYTES, consts_t=C_BYTES, state_t=STATE_BYTES, stop=99):
    """The linear layer's part 0 on the MoE path: ln -> qkv | z -> glue -> DeltaNet -> post ->
    out -> ln (+residual) -> router. `stop` is lx.py's LX_STOP debug truncation (1: after the
    glue, 2: after DeltaNet)."""
    QKV_PC, Z_PC, OUT_PC, VW, OUT_K = D.QKV_PC, D.Z_PC, D.OUT_PC, D.VW, D.OUT_K
    BB_HID, BB_OUT = X.role_band_bytes("linear", HID), X.role_band_bytes("linear_out", OUT_K)
    # 1. layer-entry norm: xn -> act[A_XN]
    tg_ln = TaskGroup()
    lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
    lni.fill(a_consts, tap=bt(consts_t, C_LNW, ELEM), wait=True, group=tg_ln)
    lno.drain(a_act, tap=bt(act_t, A_XN, ELEM), wait=True, group=tg_ln)
    # 2. qkv | z GEMV: weights now, x after the norm
    pw, py = Pipeline(3), Pipeline(3)
    for c in range(N_CORES):
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
        py.drain(y_conss[c], a_act, bt(act_t, A_QKV + c * QKV_PC * YB, QKV_PC * YB))
        py.drain(y_conss[c], a_act, bt(act_t, A_Z + c * Z_PC * YB, Z_PC * YB))
    tg_ln.finish()                                   # xn is in DDR
    px = Pipeline(3)
    px.fill(x_prod, a_act, bt(act_t, A_XN, ELEM))
    tg_s = TaskGroup()
    side_p.fill(a_act, tap=bt(act_t, A_XN, ELEM), wait=True, group=tg_s)
    side_p.fill(a_consts, tap=bt(consts_t, C_SIDE, GLUE_SIDE_BYTES), wait=True, group=tg_s)
    # qkv is in DDR. Only qkv: the glue reads A_QKV and nothing else, and every core
    # computes its 16 qkv bands before its 8 z bands, so the glue now runs while the
    # cores are still on z. z is first read by post, behind the py.finish() below.
    py.finish_oldest(*y_conss)
    # 3. glue: conv state updated in place, DeltaNet records -> act[A_VEC]
    pipe = Pipeline(3)
    for tt in range(NT):
        pipe.drain(gout_c, a_state, rows3(tt, state_t))
        if tt >= KEY_TILES:
            pipe.drain(gout_c, a_act, bt(act_t, A_VEC + (tt - KEY_TILES) * D.HEADS_PER_TILE * D.RECORD_BYTES,
                                         D.HEADS_PER_TILE * D.RECORD_BYTES))
        pipe.fill(gact_p, a_act, bt(act_t, A_QKV + tt * TILE * 4, TILE * 4))
        pipe.fill(gact_p, a_state, rows3(tt, state_t))
    pipe.finish()                                    # the records are in DDR
    tg_s.finish()
    if stop == 1:
        pw.finish()
        px.finish()
        return
    # 4. DeltaNet on the main cores: S in place, o -> act[A_O]
    X.dn_sequence(pw, py, a_state, a_act, w_prods, y_conss, act_t, A_VEC, A_O, state_t, STATE_S_OFF, S_HEAD_BYTES)
    py.finish()                                      # o is in DDR
    if stop == 2:
        pw.finish()
        px.finish()
        return
    # 6a. out projection's weights and result drains, issued BEFORE post: they read
    # consts and depend on nothing post writes. Every DeltaNet fill and drain has
    # completed by now (o is the last thing each core emits), so the throttle's waits
    # here are already satisfied; each core's w fifo fills with its first out elements
    # while post runs, instead of after.
    for c in range(N_CORES):
        pw.fill(w_prods[c], a_consts, bt(consts_t, C_WOUT + c * OUT_PC * BB_OUT, OUT_PC * BB_OUT))
        py.drain(y_conss[c], a_act, bt(act_t, A_OUT + c * OUT_PC * YB, OUT_PC * YB))
    # 5. post: og -> act[A_OG] (z from act, o from DeltaNet)
    pipe = Pipeline(3)
    pipe.fill(pin_p, a_consts, bt(consts_t, C_NW, ELEM))
    for g in range(NG):
        pipe.drain(pout_c, a_act, bt(act_t, A_OG + g * G * 2, G * 2))
        pipe.fill(pin_p, a_act, bt(act_t, A_O + g * G * 4, G * 4))
        pipe.fill(pin_p, a_act, bt(act_t, A_Z + g * G * 4, G * 4))
    pipe.finish()                                    # og is in DDR
    # 6b. out projection against og
    px.fill(x_prod, a_act, bt(act_t, A_OG, VW * 2))
    py.finish()                                      # out is in DDR
    # 7. residual + post-attention norm, then the router
    tg_ln = TaskGroup()
    lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
    lni.fill(a_consts, tap=bt(consts_t, C_POSTLN, ELEM), wait=True, group=tg_ln)
    lni.fill(a_act, tap=bt(act_t, A_OUT, HID * 4), wait=True, group=tg_ln)
    lno.drain(a_act, tap=bt(act_t, A_RES, HID * 4), wait=True, group=tg_ln)
    lno.drain(a_act, tap=bt(act_t, A_XM, ELEM), wait=True, group=tg_ln)
    tg_ln.finish()
    tg_r = TaskGroup()
    lni.fill(a_consts, tap=bt(consts_t, C_RW, X.W_ELEMS * ELEM), wait=True, group=tg_r)
    lno.drain(a_act, tap=bt(act_t, A_ROUT, ELEM), wait=True, group=tg_r)
    tg_r.finish()
    pw.finish()
    px.finish()


# ---- the full-attention layer's attention cores (Tile(2 + c, 3), c < ACORES)
A = R.attn
if A is not None:
    NH, KVH, HD = A.NH, A.KVH, A.HD
    QW, KVW = A.QW, A.KVW
    from recipes.attnknobs import probe_env  # noqa: E402
    ATTN_FLAGS = [f"-DATTN_NH={NH}", f"-DATTN_KVH={KVH}", f"-DATTN_HD={HD}", f"-DATTN_ROT={A.ROT}", "-DATTN_GATE=1",
                  f"-DATTN_VEXP={A.VEXP}", f"-DATTN_NHL={A.NHL}"]
    if A.RB > 1:                                           # attn.h defaults it to 1; adding the flag
        ATTN_FLAGS.append(f"-DATTN_RB={A.RB}")             # would change every other family's build line
    if A.BLOCK:                                            # the single-row path retired: see attn.h
        ATTN_FLAGS.append("-DATTN_BLOCK_ONLY=1")           # only this design sets it: designs/dense/dx.py
    for _k, _v in probe_env().items():                     # ATTN_NULL / ATTN_ABL: see attn.h. In the build key.
        if _k not in ("ATTN_RB", "ATTN_FAST"):             # RB is in the flags above via A.RB; FAST picks A itself.
            ATTN_FLAGS.append(f"-D{_k}={_v}")
    # The fast attention path (recipes/attnknobs.py, the same split designs/dense/dx.py
    # builds): ACORES cores at Tile(2 + c, 3), each owning NHL heads and draining its own og
    # element(s); RB cached rows per kernel call. At the defaults (1, NH, 1) every shape below
    # is the one this design always had, and the 27B / 35B kernels compile byte for byte.
    ACORES, NHL, RB = A.ACORES, A.NHL, A.RB
    # BLOCK: every cached row AND the new position's row go through attn_stepb, so attn_step /
    # attn_step_new are not built at all. That is what buys the block kernel its room on a core
    # that is also carrying the output gate's exponential and reciprocal (the 35B: a plain
    # ATTN_RB=2 overflows 16 KB of program memory on its own). recipes/attnknobs.py decides.
    BLOCK = bool(A.BLOCK)
    N_OG = NHL // A.HPO          # og elements one core emits (all of them when ACORES == 1)


def attn_types():
    """The attention core's own shapes. On the fast path q is pre-split as a bf16 [hi | lo]
    pair, ml sits at the padded stride, the accumulator covers this core's heads only and
    the parameter block grows for the row blocks; at the defaults every one of these is
    the type it always was (f32[QW], f32[2 NH], f32[QW], i32[4])."""
    a = {}
    a["u8_1k"] = np.ndarray[(A.E_A,), np.dtype[np.uint8]]    # one cache-row half: HPE f32 heads in, HPO bf16 og out
    a["b256"] = np.ndarray[(HD,), np.dtype[bfloat16]]
    a["b512"] = np.ndarray[(KVW,), np.dtype[bfloat16]]
    a["f64"] = np.ndarray[(A.ROT,), np.dtype[np.float32]]           # cos | sin of the rotated dims
    a["f256"] = np.ndarray[(HD,), np.dtype[np.float32]]
    f4096 = np.ndarray[(QW,), np.dtype[np.float32]]
    a["fq"] = np.ndarray[(2 * QW,), np.dtype[bfloat16]] if A.VEXP else f4096
    a["fml"] = np.ndarray[(2 * A.MLS,), np.dtype[np.float32]]
    a["foacc"] = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    a["pb"] = np.ndarray[(8 if RB > 1 else 4,), np.dtype[np.int32]]
    return a


def attn_kernels(inc, a):
    """The attention core's kernels, as the list the worker bodies take (`afns`)."""
    u8_1k, b256, b512, f64, f256 = a["u8_1k"], a["b256"], a["b512"], a["f64"], a["f256"]
    fq, fml, foacc, pb_ty = a["fq"], a["fml"], a["foacc"], a["pb"]

    def af(sym, args):
        return ExternalFunction(sym, source_file=str(ATTN / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=ATTN_FLAGS)

    h0_arg = [np.int32] if ACORES > 1 else []                  # only a split needs the head offset
    f_meta = af("attn_meta", [u8_1k, u8_1k, b256, b256, f64, pb_ty])
    f_q = af("attn_q", [u8_1k, b256, f64, fq, np.int32])
    f_k = af("attn_k", [u8_1k, b256, f64, f256, b512, np.int32])
    f_v = af("attn_v", [u8_1k, b512, np.int32])
    f_init = af("attn_init", [foacc, fml])
    f_step = af("attn_step", [u8_1k, u8_1k, fq, foacc, fml, pb_ty] + h0_arg) if not BLOCK else None
    f_stepn = af("attn_step_new", [b512, b512, fq, foacc, fml] + h0_arg) if not BLOCK else None
    f_stepb = af("attn_stepb", [u8_1k] * (2 * RB) + [fq, foacc, fml, pb_ty] + h0_arg) if RB > 1 else None
    # The peeled last block: RB - 1 fifo rows and the new position's k'/v' from core scratch.
    f_stepbn = (af("attn_stepb_new", [u8_1k] * (2 * (RB - 1)) + [b512, b512, fq, foacc, fml, pb_ty] + h0_arg)
                if BLOCK else None)
    f_fin = af("attn_fin", [foacc, fml, u8_1k, u8_1k, b512, np.int32])
    return ([f_meta, f_q, f_k, f_v, f_init, f_fin, f_stepb, f_stepbn] if BLOCK else
            [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin] + ([f_stepb] if RB > 1 else []))


def attn_fifos(a):
    of_ain = ObjectFifo(a["u8_1k"], name="ain", depth=max(4, 2 * RB + 2))   # a block is acquired at once
    of_aout = ObjectFifo(a["b512"], name="aout", depth=2)
    # Attention over ACORES cores: heads are independent, so each core owns NHL of them
    # and drains its own og element(s) -- the pattern dx.py uses.
    of_og = [ObjectFifo(a["b512"], name=f"og{c}", depth=2) for c in range(1, ACORES)]
    return of_ain, of_aout, of_og


def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
          fm, fq, fk, fv, fi, fs, fsn, ff, fsb, fsbn, c):
    h0 = c * NHL
    e = ain.acquire(2)                                      # [qn | kn], the position record
    fm(e[0], e[1], qn, kn, cs, pb)
    ain.release(2)
    for h in range_(A.Q_AIN_ELEMS):
        e = ain.acquire(1)
        fq(e, qn, cs, qs, h)
        ain.release(1)
    for h in range_(A.K_AIN_ELEMS):
        e = ain.acquire(1)
        fk(e, kn, cs, tmp, kout, h)
        ain.release(1)
    for h in range_(A.K_AIN_ELEMS):
        e = ain.acquire(1)
        fv(e, vout, h)
        ain.release(1)
    if aout is not None:                                    # core 0 owns the cache row
        o = aout.acquire(1)
        for j in range_(KVW):
            o[j] = kout[j]
        aout.release(1)
        o = aout.acquire(1)
        for j in range_(KVW):
            o[j] = vout[j]
        aout.release(1)
    fi(oacc, ml)
    if BLOCK:
        # pb[4] full blocks off the fifo, then ONE peeled block of RB - 1 fifo rows and
        # the new position's row. The host streams exactly RB * (pb[4] + 1) - 1 cached
        # rows (stream_patch::attn_apply), so the two counts cannot drift apart: the
        # padding rows between `pos` and the end of the last block are real transfers
        # the kernel masks. There is no single-row path left to fall back on.
        for _ in range_(pb[4]):
            e = ain.acquire(2 * RB)
            args = [e[i] for i in range(2 * RB)] + [qs, oacc, ml, pb] + ([h0] if ACORES > 1 else [])
            fsb(*args)
            ain.release(2 * RB)
        e = ain.acquire(2 * (RB - 1))
        args = [e[i] for i in range(2 * (RB - 1))] + [kout, vout, qs, oacc, ml, pb] + \
               ([h0] if ACORES > 1 else [])
        fsbn(*args)
        ain.release(2 * (RB - 1))
    elif RB > 1:
        for _ in range_(pb[4]):                             # whole blocks of RB rows
            e = ain.acquire(2 * RB)
            args = [e[i] for i in range(2 * RB)] + [qs, oacc, ml, pb] + ([h0] if ACORES > 1 else [])
            fsb(*args)
            ain.release(2 * RB)
        for _ in range_(pb[5]):                             # what did not fill a block
            e = ain.acquire(2)
            fs(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else fs(e[0], e[1], qs, oacc, ml, pb)
            ain.release(2)
        fsn(kout, vout, qs, oacc, ml, h0) if ACORES > 1 else fsn(kout, vout, qs, oacc, ml)
    else:
        for _ in range_(pb[1]):                             # nf cached rows (K_t, V_t)
            e = ain.acquire(2)
            fs(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else fs(e[0], e[1], qs, oacc, ml, pb)
            ain.release(2)
        fsn(kout, vout, qs, oacc, ml, h0) if ACORES > 1 else fsn(kout, vout, qs, oacc, ml)
    # The gate arrives for EVERY og element on the broadcast stream (two elements per
    # og element, in head order), so each core consumes all of them and computes only
    # its own N_OG. The pass-through counts are Python constants: nothing to trace.
    for _ in range(c * N_OG):
        ain.acquire(2)
        ain.release(2)
    for hp in range_(N_OG):
        g = ain.acquire(2)
        o = ogout.acquire(1)
        ff(oacc, ml, g[0], g[1], o, hp)
        ogout.release(1)
        ain.release(2)
    for _ in range((ACORES - 1 - c) * N_OG):
        ain.acquire(2)
        ain.release(2)


# Three shapes of worker body, not one with defaulted arguments: a family that does
# not block must present IRON the exact function it presented before, and the
# block-only path presents no single-row kernel at all.
def _attn_bodies():
    if BLOCK:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, ff, fsb, fsbn):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, None, None, ff,
                  fsb, fsbn, 0)

        def make_attn_body(c):
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, ff, fsb, fsbn):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, None, None,
                      ff, fsb, fsbn, c)
            return body
    elif RB > 1:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, fsb):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, fsb,
                  None, 0)

        def make_attn_body(c):
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, fsb):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff,
                      fsb, None, c)
            return body
    else:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, None,
                  None, 0)

        def make_attn_body(c):
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff,
                      None, None, c)
            return body
    return attn_body, make_attn_body


def attn_workers(of_ain, of_aout, of_og, a, afns):
    """Core 0 at Tile(2, 3) owns the cache row (aout) and og heads 0..NHL; the rest take the
    same broadcast stream in and drain their own og."""
    attn_body, make_attn_body = _attn_bodies()

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(a["b256"], name=f"qn{s}"), Buffer(a["b256"], name=f"kn{s}"), Buffer(a["f64"], name=f"cs{s}"),
                Buffer(a["fq"], name=f"qs{s}"), Buffer(a["f256"], name=f"tmp{s}"), Buffer(a["b512"], name=f"kout{s}"),
                Buffer(a["b512"], name=f"vout{s}"), Buffer(a["foacc"], name=f"oacc{s}"),
                Buffer(a["fml"], name=f"ml{s}"), Buffer(a["pb"], name=f"pb{s}")]

    workers = [Worker(attn_body, fn_args=[of_ain.cons(), of_aout.prod()] + abufs(0) + afns,
                      tile=Tile(2, 3), stack_size=STACK)]
    for c in range(1, ACORES):
        workers.append(Worker(make_attn_body(c), fn_args=[of_ain.cons(), of_og[c - 1].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=STACK))
    return workers


def w_regions(c):
    """A main core's pool regions of the full layer: q, gate, k, v, o."""
    Q_PC, KV_PC, O_PC = A.Q_PC, A.KV_PC, A.O_PC
    BB_HID, BB_O = X.role_band_bytes("attn", HID), X.role_band_bytes("attn", A.O_K)
    return [(POOL_Q + c * Q_PC * BB_HID, Q_PC * BB_HID), (POOL_GATE + c * Q_PC * BB_HID, Q_PC * BB_HID),
            (POOL_K + c * KV_PC * BB_HID, KV_PC * BB_HID), (POOL_V + c * KV_PC * BB_HID, KV_PC * BB_HID),
            (POOL_O + c * O_PC * BB_O, O_PC * BB_O)]


def y_regions(c):
    """... and its act regions: q, gate, k, v, out."""
    Q_PC, KV_PC, O_PC = A.Q_PC, A.KV_PC, A.O_PC
    qb, kb = Q_PC * YB, KV_PC * YB
    return [(AA_QG + c * qb, qb), (AA_QG + QW * 4 + c * qb, qb),
            (AA_KVN + c * kb, kb), (AA_KVN + KVW * 4 + c * kb, kb),
            (AA_OUT + c * O_PC * YB, O_PC * YB)]


def full_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                  ain_p, aout_c, og_cs, act_t=AA_BYTES, consts_t=CA_BYTES, kv_t=KV_BYTES):
    """The full-attention layer's part 0 on the MoE path: ln -> q | gate | k | v -> attention
    -> o -> ln (+residual) -> router."""
    tg_ln = TaskGroup()
    lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
    lni.fill(a_consts, tap=bt(consts_t, CA_LNW, ELEM), wait=True, group=tg_ln)
    lno.drain(a_act, tap=bt(act_t, AA_XN, ELEM), wait=True, group=tg_ln)
    pw, py = Pipeline(3), Pipeline(3)
    for c in range(N_CORES):
        for off, n in w_regions(c)[:3]:
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, off, n))
        for off, n in y_regions(c)[:3]:
            py.drain(y_conss[c], a_act, bt(act_t, off, n))
    tg_ln.finish()                                            # xn is in DDR
    tg_x = TaskGroup()
    x_prod.fill(a_act, tap=bt(act_t, AA_XN, ELEM), wait=True, group=tg_x)
    for c in range(N_CORES):
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[3]))
        py.drain(y_conss[c], a_act, bt(act_t, *y_regions(c)[3]))
    pa_out, pa_in = Pipeline(3), Pipeline(3)
    pa_out.drain(aout_c, a_kv, bt(kv_t, KV_ROW, KV_ROW))        # the new row [k' | v'] -> row pos (attnpos)
    pa_out.drain(aout_c, a_act, bt(act_t, AA_OG, NHL * HD * 2))
    for c in range(1, ACORES):                                  # heads NHL*c ..
        pa_out.drain(og_cs[c - 1], a_act, bt(act_t, AA_OG + c * NHL * HD * 2, NHL * HD * 2))
    pa_in.fill(ain_p, a_consts, bt(consts_t, CA_META, A.E_A))          # [qn | kn], one element
    pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, PTAB_ROW, PTAB_ROW))   # the position record (attnpos)
    # q is in DDR already: issuing each core's 4th drain (v) made the throttle await
    # its oldest, the q drain. The attention core takes q first (Q_AIN_ELEMS elements)
    # and k, v, the window and the gate after, so q goes now -- the attention's q stage
    # runs while the main cores are still on gate | k | v -- and the rest after them.
    assert all(len(py._q(e)) == 3 for e in y_conss), "q's drain must be the one retired"
    pa_in.fill(ain_p, a_act, bt(act_t, AA_QG, QW * 4))
    py.finish(*y_conss)                                       # gate, k, v are in DDR
    pa_in.fill(ain_p, a_act, bt(act_t, AA_KVN, KVW * 4))
    pa_in.fill(ain_p, a_act, bt(act_t, AA_KVN + KVW * 4, KVW * 4))
    pa_in.fill(ain_p, a_kv, bt(kv_t, 0, KV_ROW))                 # the window: rows [0, nf) (attnpos)
    pa_in.fill(ain_p, a_act, bt(act_t, AA_QG + QW * 4, QW * 4))
    for c in range(N_CORES):
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[4]))
        py.drain(y_conss[c], a_act, bt(act_t, *y_regions(c)[4]))
    tg_ln2 = TaskGroup()
    lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
    lni.fill(a_consts, tap=bt(consts_t, CA_POSTLN, ELEM), wait=True, group=tg_ln2)
    lno.drain(a_act, tap=bt(act_t, AA_RES, HID * 4), wait=True, group=tg_ln2)
    lno.drain(a_act, tap=bt(act_t, AA_XM, ELEM), wait=True, group=tg_ln2)
    pa_out.finish()                                           # og (and the new cache rows) are in DDR
    x_prod.fill(a_act, tap=bt(act_t, AA_OG, QW * 2), wait=True, group=tg_x)
    py.finish()                                               # out is in DDR
    lni.fill(a_act, tap=bt(act_t, AA_OUT, HID * 4), wait=True, group=tg_ln2)
    tg_r = TaskGroup()
    lni.fill(a_consts, tap=bt(consts_t, CA_RW, X.W_ELEMS * ELEM), wait=True, group=tg_r)
    lno.drain(a_act, tap=bt(act_t, AA_ROUT, ELEM), wait=True, group=tg_r)
    tg_ln2.finish()
    tg_r.finish()
    pw.finish()
    pa_in.finish()
    tg_x.finish()
