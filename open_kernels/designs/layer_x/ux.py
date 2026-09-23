r"""ux: BOTH whole-layer designs -- the linear-attention layer (lx.py) and the
full-attention layer (ax.py) -- on ONE xclbin, so the decode layer loop runs in
one `xrt::hw_context` instead of switching between two 22 times a step.

    part 0 = lx's part 0   (ln -> qkv|z -> glue -> DeltaNet -> post -> out -> ln -> router)
    part 1 = lx's MoE      (the routed + shared experts)
    part 2 = ax's part 0   (ln -> q|gate|k|v -> attention -> o -> ln -> router)
    part 3 = ax's MoE      (byte-for-byte the same core program as part 1)

The four are four CompileTime `part` values, hence four instruction streams
over one image -- the same trick lx.py already plays with its three. The
manifest names them lx0 / lx1 / ax0 / ax1 and points all four at one context.

WHY IT FITS (the main cores' 16 KB of program memory)
-----------------------------------------------------
The two layer types' main-core stage sequences are the SAME sequence with two
numbers changed:

    prep(xn) -> GEMV nA bands of K=HID -> [DeltaNet, nh heads] ->
    prep(og, 2 elems) -> GEMV 4 bands of K=4096 -> MoE

    linear:  nA = QKV_PC + Z_PC      = 24,  nh = DN_HEADS_PC = 4
    full:    nA = 2*Q_PC + 2*KV_PC   = 18,  nh = 0

Everything else already matches on this family: both og projections are 4 bands
of K = 4096 arriving in two 4 KB x elements, the MoE half is `xcommon.moe_body`
in both, and with no q8 role both GEMV stages are the same `gemv_q4_gy` call.
So the merged core program is lx's program with nA and nh read at runtime --
NOT the union of two programs. Nothing is duplicated; nothing is unrolled twice.

The two numbers arrive as RTP words: a two-int32 `use_write_rtp` Buffer per main
core, written inline by the instruction stream (parts 0 and 2 only), exactly the
plumbing designs/gemm_q4_prefill/gemm_q4_prefill.py uses to keep every K on one
image. Costs a memref.load each and 8 B of L1; costs the stream 16 write32 ops.
`nh = 0` makes `range_` run the DeltaNet loop zero times, so a full-attention
layer streams no records, no S and no S' -- the skip is a trip count, not a
branch.

THE LOOP STRUCTURE (the LX_STOP lesson)
---------------------------------------
A core's body is wrapped in while(true) and it cannot see a dispatch boundary:
it blocks on the next fifo element. One pass of a main core's body is ONE WHOLE
LAYER of EITHER type -- parts 0 and 1 together, or parts 2 and 3 together -- and
the body ends parked on the same `xin.acquire(1)` (the layer-entry norm output)
it started on. The engine never runs a part 0 without its part 1, so the core is
always at that wait point between layers, whichever type ran last.

The RTP read is ordered by that same acquire: the core acquires the xn element
BEFORE reading the words, and the runtime issues the xn fill after the writes,
so the fill cannot land -- and the acquire cannot complete -- until the words
are in place. (gemm_q4_prefill.py documents the same ordering, and why a
WorkerRuntimeBarrier is the wrong tool here.) The words a core reads are the
ones its own dispatch wrote: part 0 of layer n+1 cannot be issued before part 1
of layer n has completed, and part 1's completion needs the core past every use.

The helper cores need no parameter at all. Each one is fed by one layer type's
dispatches and simply blocks through the other's:

  (0, 3) ln + router   both types, and `xcommon.ln_router_body` is already
                       identical in lx.py and ax.py -- same elements, same order
  (1, 3) post          linear only; parked on `pin` through a full layer
  (2, 4) glue          linear only; parked on `side` through a full layer
  (2..5, 3) attention  full only; parked on `ain` through a linear layer

The glue core is the one piece that MOVED: lx.py puts it at Tile(2, 3), which is
ax.py's attention core 0. NPU2 has four compute rows and rows 4 and 5 are empty
in both designs, so it goes to Tile(2, 4) and the attention cores keep the tiles
Track A tunes them on.

SHIM BUDGET (2 fills + 2 drains per shim tile, 16 + 16 over the array)
  fills  14: lni+w0 | x+w1 | side+w2 | gact+w3 | pin+w4 | ain+w5 | w6 | w7
  drains 15: lno+y0 | pout+y1 | gout+y2 | og0+y3 | og1+y4 | og2+y5 | aout+y6 | y7
`ain` (ax: column 2) and `aout` (ax: column 1) are the two endpoints that move,
because column 2 already fills `side` and column 1 already drains `pout`.

Args are the attention layer's six (pool, xres, consts, state, act, ptab) for
every part -- one image, one kernel signature. A linear stream never touches
ptab; `state` is the DeltaNet state there and the KV cache in a full layer, so
the declared type is the larger of the two (a tap's `total` only bounds it).

WHAT THIS FILE DUPLICATES, AND WHY
The glue and post worker bodies are lx.py's and the attention worker bodies are
ax.py's, copied rather than imported: they are closures inside those designs'
`@iron.jit` functions, so there is nothing to import without editing lx.py and
ax.py -- which would move every other family's compiled bytes. **A change to
either design's helper bodies has to be mirrored here.** Factoring the three
into xcommon.py is the right move the first time both a merged and an unmerged
image have to ship; until then this note is the link.

Build: for p in 0 1 2 3: UX_PART=$p python build_design.py designs/layer_x/ux.py designs/layer_x/build_ux$p
or, with the manifest, OPEN_LAYER_ONE_CTX=1 python export_qwen36_kernels.py --only lx0,lx1,ax0,ax1 --force
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
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_HP, A_O, A_OG, A_OUT, A_QKV, A_RES, A_ROUT, A_VEC, A_XM, A_XN,  # noqa: E402
                    A_Z, AA_BYTES, AA_HP, AA_KVN, AA_OG, AA_OUT, AA_QG, AA_RES, AA_ROUT, AA_XM, AA_XN,
                    C_BYTES, C_LNW, C_NW, C_POSTLN, C_RW, C_SGW, C_SIDE, C_WOUT, CA_BYTES, CA_LNW,
                    CA_META, CA_POSTLN, CA_RW, CA_SGW, GLUE_SIDE_BYTES, KV_BYTES, KV_ROW, POOL_BYTES,
                    POOL_GATE, POOL_K, POOL_O, POOL_Q, POOL_QKV, POOL_V, POOL_Z, PTAB_BYTES, PTAB_ROW,
                    STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES, R, SPEC)
import xcommon as X  # noqa: E402

D = R.linear
A = R.attn
if D is None or A is None:
    sys.exit("ux.py: the merged layer image needs BOTH a linear-attention and a full-attention layer type")
if X.KIND != "moe":
    sys.exit("ux.py: the merged layer image is the MoE composition's (qwen36moe); the dense tail is lx/ax")
if X.Q8:
    sys.exit(f"ux.py: a q8 projection role ({sorted(X.Q8)}) would put two GEMV entries on the main core; "
             "the merged image assumes both stages take the same q4_1 call")

HID = X.HID
N_CORES = X.N_CORES
ELEM = X.ELEM
# The two stages the layer types share exactly. Asserted, not assumed: if a spec ever moves
# the og projection apart the merged program needs a third RTP word, and this is where it says so.
QKV_PC, Z_PC, OUT_PC, VW, OUT_K = D.QKV_PC, D.Z_PC, D.OUT_PC, D.VW, D.OUT_K
Q_PC, KV_PC, O_PC, QW, KVW, O_K = A.Q_PC, A.KV_PC, A.O_PC, A.QW, A.KVW, A.O_K
NH, KVH, HD = A.NH, A.KVH, A.HD
if (OUT_PC, OUT_K, D.OG_ELEMS) != (O_PC, O_K, A.OG_ELEMS):
    sys.exit(f"ux.py: the two og projections differ -- linear {OUT_PC} bands of K={OUT_K} in "
             f"{D.OG_ELEMS} elements against full {O_PC} of K={O_K} in {A.OG_ELEMS}. One image "
             "can only share that stage while they match.")
OG_ELEMS = D.OG_ELEMS
NBANDS_LIN = QKV_PC + Z_PC                       # RTP word 0, linear stream
NBANDS_FULL = 2 * Q_PC + 2 * KV_PC               # RTP word 0, full stream
DN_HEADS_PC = X.DN_HEADS_PC                      # RTP word 1, linear stream (full: 0)

# dn_glue / dn_post (lx.py's constants, verbatim)
NCH, NHEAD = D.NCH, D.NHEAD
TILE, NT = D.TILE, D.NT
AB_ELEMS = D.AB_ELEMS
G, NG = D.G, D.NG
CONV_ROWS = SPEC.conv_kernel - 1
KEY_TILES = D.VALUE_TILE0
VALUE_TILES = NT - KEY_TILES
CONVW_ELEMS = SPEC.conv_kernel * TILE * 2 // ELEM
GLUE_NHEAD_DEFAULT = 32
GLUE_FLAGS = {} if NHEAD == GLUE_NHEAD_DEFAULT else {"compile_flags": [f"-DDNGLUE_NHEAD={NHEAD}"]}

# attn (ax.py's, verbatim: Track A owns these)
from recipes.attnknobs import probe_env  # noqa: E402
ATTN_FLAGS = [f"-DATTN_NH={NH}", f"-DATTN_KVH={KVH}", f"-DATTN_HD={HD}", f"-DATTN_ROT={A.ROT}", "-DATTN_GATE=1",
              f"-DATTN_VEXP={A.VEXP}", f"-DATTN_NHL={A.NHL}"]
if A.RB > 1:
    ATTN_FLAGS.append(f"-DATTN_RB={A.RB}")
if A.BLOCK:                                      # the single-row path retired: see attn.h
    ATTN_FLAGS.append("-DATTN_BLOCK_ONLY=1")
for _k, _v in probe_env().items():
    if _k not in ("ATTN_RB", "ATTN_FAST"):
        ATTN_FLAGS.append(f"-D{_k}={_v}")
ACORES, NHL, RB = A.ACORES, A.NHL, A.RB
BLOCK = bool(A.BLOCK)                            # every row through attn_stepb (ax.py's BLOCK)

PART = int(os.environ.get("UX_PART", 0))         # 0 lx0 | 1 lx1 | 2 ax0 | 3 ax1
if PART not in (0, 1, 2, 3):
    sys.exit(f"ux.py: UX_PART={PART} (0 = lx0, 1 = lx1, 2 = ax0, 3 = ax1)")

# ONE set of buffer-argument shapes for all four streams, so the four builds emit the same
# image. A tap's `total` only bounds its offsets, and every linear offset is inside the
# larger attention buffer and vice versa, so widening the declaration changes no BD.
CONSTS_T = max(C_BYTES, CA_BYTES)
ACT_T = max(A_BYTES, AA_BYTES)
STATE_T = max(STATE_BYTES, KV_BYTES)

GLUE_TILE = Tile(2, 4)          # lx.py's Tile(2, 3) is ax.py's attention core 0; rows 4-5 are empty


def rows3(t: int):
    """Tile t of each conv-state row, in BYTES of the state BO (lx.py's, against STATE_T)."""
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, STATE_T), t * TILE * 2, [1, 1, CONV_ROWS, TILE * 2], [0, 0, NCH * 2, 1])


def dn_body_n(win, yout, B, K, nheads):
    """xcommon.dn_body with the head count read from the stream instead of the recipe:
    a full-attention layer passes 0 and the loop runs zero times."""
    ds = B["ds"]
    for _ in range_(nheads):
        re_ = win.acquire(1)
        K["vcopy"](re_, ds)
        win.release(1)
        for blk in range_(X.DN_SLICES):
            se = win.acquire(1)
            K["p1"](se, ds, blk)
            win.release(1)
        K["delta"](ds)
        for blk in range_(X.DN_SLICES):
            se = win.acquire(1)
            for j in range_(2 * X.DN_ROWS):
                ye = yout.acquire(1)
                K["row"](se, ds, ye, blk, j)
                yout.release(1)
            win.release(1)
        for hf in range_(2):
            ye = yout.acquire(1)
            K["ofin"](ds, ye, hf)
            yout.release(1)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ux(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
       srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    u8_4k = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    u8_2k = np.ndarray[(2048,), np.dtype[np.uint8]]
    u8_1k = np.ndarray[(A.E_A,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CONSTS_T,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_T,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(ACT_T,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]
    nw_ty = np.ndarray[(SPEC.lin_value_dim,), np.dtype[bfloat16]]
    f32 = np.ndarray[(NHEAD,), np.dtype[np.float32]]
    fqk = np.ndarray[(2 * D.KEY_WIDTH,), np.dtype[np.float32]]
    fvt = np.ndarray[(TILE,), np.dtype[np.float32]]
    fxn = np.ndarray[(HID,), np.dtype[bfloat16]]
    b256 = np.ndarray[(HD,), np.dtype[bfloat16]]
    b512 = np.ndarray[(KVW,), np.dtype[bfloat16]]
    f64 = np.ndarray[(A.ROT,), np.dtype[np.float32]]
    f256 = np.ndarray[(HD,), np.dtype[np.float32]]
    f4096 = np.ndarray[(QW,), np.dtype[np.float32]]
    fq = np.ndarray[(2 * QW,), np.dtype[bfloat16]] if A.VEXP else f4096
    fml = np.ndarray[(2 * A.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    pb_ty = np.ndarray[(8 if RB > 1 else 4,), np.dtype[np.int32]]

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(ATTN), str(X.LN), str(X.LINL), str(X.RT),
                            str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    f_ab = ExternalFunction("glue_ab", source_file=str(GLUE / "glue_ab.cc"), arg_types=[u8_4k, fxn, f32, np.int32],
                            include_dirs=inc, **GLUE_FLAGS)
    f_small = ExternalFunction("glue_small_fn", source_file=str(GLUE / "glue_small.cc"),
                               arg_types=[u8_4k, f32, f32, f32, f32], include_dirs=inc, **GLUE_FLAGS)
    f_conv = ExternalFunction("glue_conv", source_file=str(GLUE / "glue_conv.cc"),
                              arg_types=[u8_2k, u8_2k, u8_2k, u8_2k, u8_2k, u8_4k, u8_4k, u8_2k, u8_2k, u8_2k, fqk, fvt,
                                         np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_emit = ExternalFunction("glue_emit_fn", source_file=str(GLUE / "glue_emit.cc"),
                              arg_types=[fqk, fvt, f32, f32, u8_2k, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_copy = ExternalFunction("glue_copy_xn", source_file=str(GLUE / "glue_copy.cc"), arg_types=[u8_4k, fxn],
                              include_dirs=inc, **GLUE_FLAGS)
    post_fn = ExternalFunction("post_fn", source_file=str(POST / "post.cc"), arg_types=[u8_4k, u8_4k, nw_ty, u8_2k],
                               include_dirs=inc)
    post_copy = ExternalFunction("post_copy_nw", source_file=str(POST / "post_copy.cc"), arg_types=[u8_4k, nw_ty],
                                 include_dirs=inc)

    def af(sym, args):
        return ExternalFunction(sym, source_file=str(ATTN / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=ATTN_FLAGS)

    h0_arg = [np.int32] if ACORES > 1 else []
    f_meta = af("attn_meta", [u8_1k, u8_1k, b256, b256, f64, pb_ty])
    f_q = af("attn_q", [u8_1k, b256, f64, fq, np.int32])
    f_k = af("attn_k", [u8_1k, b256, f64, f256, b512, np.int32])
    f_v = af("attn_v", [u8_1k, b512, np.int32])
    f_init = af("attn_init", [foacc, fml])
    f_step = af("attn_step", [u8_1k, u8_1k, fq, foacc, fml, pb_ty] + h0_arg) if not BLOCK else None
    f_stepn = af("attn_step_new", [b512, b512, fq, foacc, fml] + h0_arg) if not BLOCK else None
    f_stepb = af("attn_stepb", [u8_1k] * (2 * RB) + [fq, foacc, fml, pb_ty] + h0_arg) if RB > 1 else None
    f_stepbn = (af("attn_stepb_new", [u8_1k] * (2 * (RB - 1)) + [b512, b512, fq, foacc, fml, pb_ty] + h0_arg)
                if BLOCK else None)
    f_fin = af("attn_fin", [foacc, fml, u8_1k, u8_1k, b512, np.int32])

    # ---- fifos: the union of lx's and ax's
    of_w = [ObjectFifo(t["elem"], name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(t["y"], name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(t["x"], name="x", depth=2)
    of_lni = ObjectFifo(u8_4k, name="lni", depth=5)
    of_lno = ObjectFifo(u8_4k, name="lno", depth=3)
    of_side = ObjectFifo(u8_4k, name="side", depth=2)
    of_gact = ObjectFifo(u8_2k, name="gact", depth=5)
    of_gout = ObjectFifo(u8_2k, name="gout", depth=3)
    of_pin = ObjectFifo(u8_4k, name="pin", depth=2)
    of_pout = ObjectFifo(u8_2k, name="pout", depth=2)
    of_ain = ObjectFifo(u8_1k, name="ain", depth=max(4, 2 * RB + 2))
    of_aout = ObjectFifo(b512, name="aout", depth=2)
    of_og = [ObjectFifo(b512, name=f"og{c}", depth=2) for c in range(1, ACORES)]

    # The per-core parameter words. Zeros in the image: the real counts belong to the
    # instruction stream, which is the whole point (gemm_q4_prefill.py's rtp_bufs).
    rtp = [Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(2, dtype=np.int32), use_write_rtp=True)
           for c in range(N_CORES)]

    # ---- cores
    def main_body(win, xin, yout, my_rtp, *args):
        B, K = X.unpack_args(args)
        tab = B["tab"]
        # The acquire comes FIRST and orders the reads: the runtime issues this fill after the
        # write32s, so the element cannot arrive before the words are in place.
        xe = xin.acquire(1)
        nbands = my_rtp[0]                                  # 24 linear | 18 full
        nheads = my_rtp[1]                                  # 4 linear | 0 full
        K["prep2048"](xe, tab)
        X.gemv_bands(win, yout, tab, K["gy"], nbands, X.n_groups(HID), X.per_band(HID), 2)
        xin.release(1)
        dn_body_n(win, yout, B, K, nheads)
        oe = xin.acquire(2)                                 # og, K = OUT_K (= O_K)
        K["prep4096a"](oe[0], tab)
        K["prep4096b"](oe[1], tab)
        X.role_gemv_bands(win, yout, B, K, "linear_out", OUT_PC, OUT_K)
        xin.release(2)
        X.moe_body(win, xin, yout, B, K)

    def glue_body(sin, ain, oout, acc_a, acc_b, decay, beta, qk, vt, xn, fab, fsmall, fconv, femit, fcopy):
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

    N_OG = NHL // A.HPO

    def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
              fm, fq, fk, fv, fi, fs, fsn, ff, fsb, fsbn, c):
        h0 = c * NHL
        e = ain.acquire(2)
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
        if aout is not None:
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
            # ax.py's block-only walk: pb[4] full blocks off the fifo, then ONE peeled block
            # of RB - 1 fifo rows and the new position's row (the host pads the streamed
            # row count to RB * (pb[4] + 1) - 1 from the manifest's ax0 `rb`).
            for _ in range_(pb[4]):
                e = ain.acquire(2 * RB)
                args = [e[i] for i in range(2 * RB)] + [qs, oacc, ml, pb] + ([h0] if ACORES > 1 else [])
                fsb(*args)
                ain.release(2 * RB)
            e = ain.acquire(2 * (RB - 1))
            args = [e[i] for i in range(2 * (RB - 1))] + [kout, vout, qs, oacc, ml, pb] +                    ([h0] if ACORES > 1 else [])
            fsbn(*args)
            ain.release(2 * (RB - 1))
        elif RB > 1:
            for _ in range_(pb[4]):
                e = ain.acquire(2 * RB)
                args = [e[i] for i in range(2 * RB)] + [qs, oacc, ml, pb] + ([h0] if ACORES > 1 else [])
                fsb(*args)
                ain.release(2 * RB)
            for _ in range_(pb[5]):
                e = ain.acquire(2)
                fs(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else fs(e[0], e[1], qs, oacc, ml, pb)
                ain.release(2)
            fsn(kout, vout, qs, oacc, ml, h0) if ACORES > 1 else fsn(kout, vout, qs, oacc, ml)
        else:
            for _ in range_(pb[1]):
                e = ain.acquire(2)
                fs(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else fs(e[0], e[1], qs, oacc, ml, pb)
                ain.release(2)
            fsn(kout, vout, qs, oacc, ml, h0) if ACORES > 1 else fsn(kout, vout, qs, oacc, ml)
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

    # ax.py's three worker-body shapes: a family that does not block presents IRON the exact
    # function it presented before, and the block-only path presents no single-row kernel.
    if BLOCK:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, ff, fsb, fsbn):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, None, None, ff, fsb, fsbn, 0)

        def make_attn_body(c):
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, ff, fsb, fsbn):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, None, None, ff, fsb, fsbn, c)
            return body
    elif RB > 1:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, fsb):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, fsb, None, 0)

        def make_attn_body(c):
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, fsb):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, fsb, None, c)
            return body
    else:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, None, None, 0)

        def make_attn_body(c):
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq, fk, fv, fi, fs, fsn, ff, None, None, c)
            return body

    workers = [Worker(X.ln_router_body,
                      fn_args=[of_lni.cons(), of_lno.prod(), Buffer(tl["xb"], name="rxs"),
                               Buffer(tl["racc"], name="racc"),
                               L["ln_nr"], L["ln"], L["rcopy"], L["racc"], L["rfin"]],
                      tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body,
                              fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), rtp[c],
                                       *X.worker_args(X.core_buffers(t, c), K)],
                              tile=Tile(c, 2), stack_size=0x1800))
    workers.append(Worker(post_body, fn_args=[of_pin.cons(), of_pout.prod(), Buffer(nw_ty, name="nwb"), post_fn, post_copy],
                          tile=Tile(1, 3), stack_size=0x1800))
    workers.append(Worker(glue_body,
                          fn_args=[of_side.cons(), of_gact.cons(), of_gout.prod(),
                                   Buffer(f32, name="acc_a"), Buffer(f32, name="acc_b"), Buffer(f32, name="decay"),
                                   Buffer(f32, name="beta"), Buffer(fqk, name="qk"), Buffer(fvt, name="vt"),
                                   Buffer(fxn, name="xnb"), f_ab, f_small, f_conv, f_emit, f_copy],
                          tile=GLUE_TILE, stack_size=0x1800))

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(b256, name=f"qn{s}"), Buffer(b256, name=f"kn{s}"), Buffer(f64, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(f256, name=f"tmp{s}"), Buffer(b512, name=f"kout{s}"),
                Buffer(b512, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]

    afns = ([f_meta, f_q, f_k, f_v, f_init, f_fin, f_stepb, f_stepbn] if BLOCK else
            [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin] + ([f_stepb] if RB > 1 else []))
    workers.append(Worker(attn_body, fn_args=[of_ain.cons(), of_aout.prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=0x1800))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn_body(c), fn_args=[of_ain.cons(), of_og[c - 1].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=0x1800))

    bt = X.bt
    BB_HID = X.role_band_bytes("linear", HID)
    BB_OUT = X.role_band_bytes("linear_out", OUT_K)
    BB_O = X.role_band_bytes("attn", O_K)
    YB = X.BAND_ROWS * 4

    def w_regions(c):
        return [(POOL_Q + c * Q_PC * BB_HID, Q_PC * BB_HID), (POOL_GATE + c * Q_PC * BB_HID, Q_PC * BB_HID),
                (POOL_K + c * KV_PC * BB_HID, KV_PC * BB_HID), (POOL_V + c * KV_PC * BB_HID, KV_PC * BB_HID),
                (POOL_O + c * O_PC * BB_O, O_PC * BB_O)]

    def y_regions(c):
        qb, kb = Q_PC * YB, KV_PC * YB
        return [(AA_QG + c * qb, qb), (AA_QG + QW * 4 + c * qb, qb),
                (AA_KVN + c * kb, kb), (AA_KVN + KVW * 4 + c * kb, kb),
                (AA_OUT + c * O_PC * YB, O_PC * YB)]

    # ---- host sequences (one per instruction stream)
    def linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                        side_p, gact_p, gout_c, pin_p, pout_c):
        """lx.py's part 0, against the merged buffer totals, plus the two RTP words."""
        for c in range(N_CORES):
            rtp[c][0] = NBANDS_LIN
            rtp[c][1] = DN_HEADS_PC
        # 1. layer-entry norm: xn -> act[A_XN]
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(CONSTS_T, C_LNW, ELEM), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(ACT_T, A_XN, ELEM), wait=True, group=tg_ln)
        # 2. qkv | z GEMV: weights now, x after the norm
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
            py.drain(y_conss[c], a_act, bt(ACT_T, A_QKV + c * QKV_PC * YB, QKV_PC * YB))
            py.drain(y_conss[c], a_act, bt(ACT_T, A_Z + c * Z_PC * YB, Z_PC * YB))
        tg_ln.finish()                                   # xn is in DDR
        px = Pipeline(3)
        px.fill(x_prod, a_act, bt(ACT_T, A_XN, ELEM))
        tg_s = TaskGroup()
        side_p.fill(a_act, tap=bt(ACT_T, A_XN, ELEM), wait=True, group=tg_s)
        side_p.fill(a_consts, tap=bt(CONSTS_T, C_SIDE, GLUE_SIDE_BYTES), wait=True, group=tg_s)
        # qkv is in DDR. Only qkv (lx.py's part 0 does the same): the glue reads A_QKV and
        # nothing else, and every core computes its qkv bands before its z bands. z is first
        # read by post, behind the py.finish() after dn_sequence.
        py.finish_oldest(*y_conss)
        # 3. glue: conv state updated in place, DeltaNet records -> act[A_VEC]
        pipe = Pipeline(3)
        for tt in range(NT):
            pipe.drain(gout_c, a_state, rows3(tt))
            if tt >= KEY_TILES:
                pipe.drain(gout_c, a_act, bt(ACT_T, A_VEC + (tt - KEY_TILES) * D.HEADS_PER_TILE * D.RECORD_BYTES,
                                             D.HEADS_PER_TILE * D.RECORD_BYTES))
            pipe.fill(gact_p, a_act, bt(ACT_T, A_QKV + tt * TILE * 4, TILE * 4))
            pipe.fill(gact_p, a_state, rows3(tt))
        pipe.finish()                                    # the records are in DDR
        tg_s.finish()
        # 4. DeltaNet on the main cores: S in place, o -> act[A_O]
        X.dn_sequence(pw, py, a_state, a_act, w_prods, y_conss, ACT_T, A_VEC, A_O, STATE_T, STATE_S_OFF, S_HEAD_BYTES)
        py.finish()                                      # o (and z) are in DDR
        # 6a. out projection's weights and result drains, issued BEFORE post (as lx.py): they
        # read consts and depend on nothing post writes, so each core's w fifo fills with its
        # first out elements while post runs. Per channel the order is unchanged.
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_consts, bt(CONSTS_T, C_WOUT + c * OUT_PC * BB_OUT, OUT_PC * BB_OUT))
            py.drain(y_conss[c], a_act, bt(ACT_T, A_OUT + c * OUT_PC * YB, OUT_PC * YB))
        # 5. post: og -> act[A_OG] (z from act, o from DeltaNet)
        pipe = Pipeline(3)
        pipe.fill(pin_p, a_consts, bt(CONSTS_T, C_NW, ELEM))
        for g in range(NG):
            pipe.drain(pout_c, a_act, bt(ACT_T, A_OG + g * G * 2, G * 2))
            pipe.fill(pin_p, a_act, bt(ACT_T, A_O + g * G * 4, G * 4))
            pipe.fill(pin_p, a_act, bt(ACT_T, A_Z + g * G * 4, G * 4))
        pipe.finish()                                    # og is in DDR
        # 6b. out projection against og
        px.fill(x_prod, a_act, bt(ACT_T, A_OG, VW * 2))
        py.finish()                                      # out is in DDR
        # 7. residual + post-attention norm, then the router
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(CONSTS_T, C_POSTLN, ELEM), wait=True, group=tg_ln)
        lni.fill(a_act, tap=bt(ACT_T, A_OUT, HID * 4), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(ACT_T, A_RES, HID * 4), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(ACT_T, A_XM, ELEM), wait=True, group=tg_ln)
        tg_ln.finish()
        tg_r = TaskGroup()
        lni.fill(a_consts, tap=bt(CONSTS_T, C_RW, X.W_ELEMS * ELEM), wait=True, group=tg_r)
        lno.drain(a_act, tap=bt(ACT_T, A_ROUT, ELEM), wait=True, group=tg_r)
        tg_r.finish()
        pw.finish()
        px.finish()

    def full_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                      ain_p, aout_c, og_cs):
        """ax.py's part 0, against the merged buffer totals, plus the two RTP words."""
        for c in range(N_CORES):
            rtp[c][0] = NBANDS_FULL
            rtp[c][1] = 0
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(CONSTS_T, CA_LNW, ELEM), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(ACT_T, AA_XN, ELEM), wait=True, group=tg_ln)
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            for off, n in w_regions(c)[:3]:
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, off, n))
            for off, n in y_regions(c)[:3]:
                py.drain(y_conss[c], a_act, bt(ACT_T, off, n))
        tg_ln.finish()                                            # xn is in DDR
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(ACT_T, AA_XN, ELEM), wait=True, group=tg_x)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[3]))
            py.drain(y_conss[c], a_act, bt(ACT_T, *y_regions(c)[3]))
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        pa_out.drain(aout_c, a_kv, bt(STATE_T, KV_ROW, KV_ROW))          # the new row [k' | v'] (attnpos)
        pa_out.drain(aout_c, a_act, bt(ACT_T, AA_OG, NHL * HD * 2))
        for c in range(1, ACORES):
            pa_out.drain(og_cs[c - 1], a_act, bt(ACT_T, AA_OG + c * NHL * HD * 2, NHL * HD * 2))
        pa_in.fill(ain_p, a_consts, bt(CONSTS_T, CA_META, A.E_A))        # [qn | kn], one element
        pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, PTAB_ROW, PTAB_ROW))    # the position record (attnpos)
        # q is in DDR already: each core's 4th drain (v) made the throttle retire its oldest,
        # the q drain. The attention core takes q first, so q goes now (as ax.py) and the
        # attention's q stage runs while the main cores are still on gate | k | v.
        assert all(len(py._q(e)) == 3 for e in y_conss), "q's drain must be the one retired"
        pa_in.fill(ain_p, a_act, bt(ACT_T, AA_QG, QW * 4))
        py.finish(*y_conss)                                       # gate, k, v are in DDR
        pa_in.fill(ain_p, a_act, bt(ACT_T, AA_KVN, KVW * 4))
        pa_in.fill(ain_p, a_act, bt(ACT_T, AA_KVN + KVW * 4, KVW * 4))
        pa_in.fill(ain_p, a_kv, bt(STATE_T, 0, KV_ROW))                  # the window: rows [0, nf) (attnpos)
        pa_in.fill(ain_p, a_act, bt(ACT_T, AA_QG + QW * 4, QW * 4))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[4]))
            py.drain(y_conss[c], a_act, bt(ACT_T, *y_regions(c)[4]))
        tg_ln2 = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(CONSTS_T, CA_POSTLN, ELEM), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(ACT_T, AA_RES, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(ACT_T, AA_XM, ELEM), wait=True, group=tg_ln2)
        pa_out.finish()                                           # og (and the new cache rows) are in DDR
        x_prod.fill(a_act, tap=bt(ACT_T, AA_OG, QW * 2), wait=True, group=tg_x)
        py.finish()                                               # out is in DDR
        lni.fill(a_act, tap=bt(ACT_T, AA_OUT, HID * 4), wait=True, group=tg_ln2)
        tg_r = TaskGroup()
        lni.fill(a_consts, tap=bt(CONSTS_T, CA_RW, X.W_ELEMS * ELEM), wait=True, group=tg_r)
        lno.drain(a_act, tap=bt(ACT_T, AA_ROUT, ELEM), wait=True, group=tg_r)
        tg_ln2.finish()
        tg_r.finish()
        pw.finish()
        pa_in.finish()
        tg_x.finish()

    def sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                 side_p, gact_p, gout_c, pin_p, pout_c, ain_p, aout_c, og_cs):
        if part == 0:
            linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                            side_p, gact_p, gout_c, pin_p, pout_c)
        elif part == 1:
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres,
                           w_prods, x_prod, y_conss, ACT_T, CONSTS_T, A_XM, A_ROUT, A_RES, A_HP, C_SGW)
        elif part == 2:
            full_sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                          ain_p, aout_c, og_cs)
        else:
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres,
                           w_prods, x_prod, y_conss, ACT_T, CONSTS_T, AA_XM, AA_ROUT, AA_RES, AA_HP, CA_SGW)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)),
                            of_gout.cons(tile=Tile(2, 0)),
                            of_pin.prod(tile=Tile(4, 0)), of_pout.cons(tile=Tile(1, 0)),
                            of_ain.prod(tile=Tile(5, 0)), of_aout.cons(tile=Tile(6, 0)),
                            [of_og[c].cons(tile=Tile(3 + c, 0)) for c in range(ACORES - 1)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = ux
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes(), (HERE / "ux.py").read_bytes()] + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.cc"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.h")) + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(),
                   (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
