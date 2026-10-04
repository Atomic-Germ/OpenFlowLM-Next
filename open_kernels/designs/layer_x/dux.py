r"""dux: the dense composition's two whole-layer designs -- lx.py's linear-attention layer and
ax.py's full-attention layer (the Qwen3.5 / Qwen3.8 dense tail) -- on ONE xclbin, so the decode
layer loop runs in one `xrt::hw_context` instead of switching between two at every layer-type
boundary (OPEN-DECODE-ONE-CONTEXT; the dense twin of the MoE family's ux.py upstream, PR #116).

    part 0 = lx.py's dense stream   (ln -> qkv|z -> glue -> DeltaNet -> post -> out -> ln -> FFN)
    part 1 = ax.py's dense stream   (ln -> q|gate|k|v -> attention -> o -> ln -> FFN)

Two CompileTime `part` values, hence two instruction streams over one image. The manifest names
them lx / ax and points both at one context, "layer", so the engine's submit-ahead (which queues
the next dispatch only inside one context) also runs across the type boundary.

WHY IT PAYS (Ternary Bonsai 2 27B, 2026-10-03, --bench-decode at position 513): a step has 32
lx <-> ax boundaries, and the dispatch after each one costs ~1.2 ms more than the same dispatch
after one of its own type -- ~39 ms of a ~285 ms step.

WHY IT FITS (the main cores' 16 KB of program memory)
-----------------------------------------------------
The two layer types' main-core stage sequences are the SAME sequence with two numbers changed:

    prep(xn, 3 elems) -> GEMV nA bands of K=HID -> [DeltaNet, nh heads] ->
    prep(og, 3 elems) -> GEMV OUT_PC bands of K=OUT_K -> the dense FFN

    linear:  nA = QKV_PC + Z_PC,      nh = DN_HEADS_PC
    full:    nA = 2 Q_PC + 2 KV_PC,   nh = 0

Asserted below: both og projections have the same band count, K and element count (the 27B:
10 bands of K = 6144 = 48 x 128 = 24 x 256), and every projection role streams through the same
GEMV entry (no q8 role, no folded mixed entry). So the merged main-core program is lx's program
with nA and nh read at run time -- not the union of two programs. The two numbers arrive as RTP
words (a three-int32 `use_write_rtp` Buffer per main core, written inline at the head of each
stream, gemm_q4_prefill.py's plumbing). `nh = 0` runs the DeltaNet loop zero times, so a full
layer streams no records, no S and no S'. The third word is the DeltaNet slice count, constant
but read at run time on purpose: with a compile-time 26 LLVM unrolled pass 1 into 26 call sites.

THE LOOP STRUCTURE AND THE RTP ORDERING (ux.py's)
-------------------------------------------------
A core's body is wrapped in while(true) and cannot see a dispatch boundary: one pass of the main
cores' body is one whole layer of either type, ending parked on the first x element of the next
layer. The words are read AFTER the xn elements are acquired: the stream writes the words first
and issues the xn fill only after the entry norm, so the fill cannot land before the words.

The helper cores need no parameter: each is fed by one layer type's dispatches and blocks
through the other's.
  (0, 3)       the norm helper   both types (the same ln_dense worker in lx.py and ax.py)
  (1, 3)       post              linear only; parked on `pin` through a full layer
  (2, 4)       glue              linear only; MOVED from lx.py's Tile(2, 3), which is attention
                                 core 0; rows 4 and 5 are empty in both designs
  (2..7, 3)    attention         full only; parked on `ain` through a linear layer

SHIM BUDGET (2 fills + 2 drains per shim tile, 16 + 16 over the array)
  fills  14: lni+w0 | x+w1 | side+w2 | gact+w3 | pin+w4 | ain+w5 | w6 | w7
  drains 13: lno+y0 | pout+y1 | gout+y2 | ogj+y3 | aout+y4 | y5 | y6 | y7
lx.py + ax.py's endpoints would need 17 drains (the 27B's attention runs on six cores, each
draining its own og element). So attention cores 1..5's og elements are JOINED in a memtile into
one element (`ogj`, one 10 KB drain to act[AA_OG + one core's og]); core 0 keeps `aout` for the
new cache row and its own og exactly as in ax.py. A join needs one element from every input per
joined element -- each attention core emits exactly N_OG = 1 og element per layer.

Args are the attention layer's six (pool, xres, consts, state, act, ptab) for both parts -- one
image, one kernel signature. The linear stream never touches ptab; `state` is the DeltaNet state
there and the KV cache in a full layer, so each buffer type is declared at the larger size (a
tap's `total` only bounds its offsets; every tap below uses the merged totals).

Build: for p in 0 1: DUX_PART=$p python build_design.py designs/layer_x/dux.py designs/layer_x/build_dux$p
or through the recipe (recipes/qwen35.py `merged_image`), which exports them as `lx` and `ax`.
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
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_H, A_O, A_OG, A_OUT, A_OUT2, A_OUT2B, A_QKV, A_RES,  # noqa: E402
                    A_VEC, A_XM, A_XN, A_Z,
                    AA_BYTES, AA_H, AA_KVN, AA_OG, AA_OUT, AA_OUT2, AA_OUT2B, AA_QG, AA_RES, AA_XM, AA_XN,
                    C_BYTES, C_LNW, C_NW, C_POSTLN, C_SIDE, C_WOUT, CA_BYTES, CA_LNW, CA_META, CA_POSTLN,
                    ELN, GLUE_SIDE_BYTES, KV_BYTES, KV_ROW, POOL_BYTES, POOL_FFN_DOWN, POOL_FFN_GATE,
                    POOL_FFN_UP, POOL_GATE, POOL_K, POOL_O, POOL_Q, POOL_QKV, POOL_V, POOL_Z, PTAB_BYTES,
                    PTAB_ROW, S_HEAD_BYTES, SIDE_ALPHA, SIDE_BETA, SIDE_CONV, SIDE_SMALL, STATE_BYTES,
                    STATE_S_OFF, R, SPEC)
import xcommon as X  # noqa: E402

D = R.linear
A = R.attn
if D is None or A is None:
    sys.exit("dux.py: the merged layer image needs BOTH a linear-attention and a full-attention layer type")
if X.KIND != "dense":
    sys.exit("dux.py: the merged dense image is the Qwen3.5 composition's; the MoE family's is ux.py")
if X.Q8 or X.MIXED:
    sys.exit(f"dux.py: a q8 projection role ({sorted(X.Q8)}) puts a second GEMV entry on the main core; "
             "the merged image assumes every first-stage GEMV is the same call")
if not X.LN_SPLIT:
    sys.exit("dux.py: only the split norm helper (recipes/qwen36moe.py norm_split) is merged here")

HID = X.HID
N_CORES = X.N_CORES
ELEM = X.ELEM
QKV_PC, Z_PC, OUT_PC, OUT_K = D.QKV_PC, D.Z_PC, D.OUT_PC, D.OUT_K
NH, KVH, HD = A.NH, A.KVH, A.HD
Q_PC, KV_PC, O_PC, O_K = A.Q_PC, A.KV_PC, A.O_PC, A.O_K
QW, KVW = A.QW, A.KVW
if (OUT_PC, OUT_K, D.OG_ELEMS) != (O_PC, O_K, A.OG_ELEMS):
    sys.exit(f"dux.py: the two og projections differ -- linear {OUT_PC} bands of K={OUT_K} in {D.OG_ELEMS} "
             f"elements against full {O_PC} of K={O_K} in {A.OG_ELEMS}; one image can share that stage only "
             "while they match")
if D.XN_SIDE_ELEMS != X.FFN.XN_ELEMS:
    sys.exit("dux.py: the two layer types receive the xn in different element counts")
XN_ELEMS = D.XN_SIDE_ELEMS
OG_ELEMS = D.OG_ELEMS
NBANDS_LIN = QKV_PC + Z_PC                       # RTP word 0, linear stream
NBANDS_FULL = 2 * Q_PC + 2 * KV_PC               # RTP word 0, full stream
DN_HEADS_PC = X.DN_HEADS_PC                      # RTP word 1, linear stream (full: 0); word 2 = DN_SLICES
if XN_ELEMS <= X.X_DEPTH or OG_ELEMS <= X.X_DEPTH:
    sys.exit("dux.py: the merged main-core program streams the xn and og element by element (prep_stream)")

# ---- the linear layer's helpers (lx.py)
NCH, NHEAD = D.NCH, D.NHEAD
TILE, NT = D.TILE, D.NT
G, NG = D.G, D.NG
CONV_ROWS = SPEC.conv_kernel - 1
KEY_TILES = D.VALUE_TILE0
VALUE_TILES = NT - KEY_TILES
CONVW_ELEMS = SPEC.conv_kernel * TILE * 2 // ELEM
GLUE_NHEAD_DEFAULT = 32
AB_TILES = [min(ELEM // 2, HID - h * (ELEM // 2)) // D.AB_ROWS for h in range(XN_ELEMS)]
assert sum(AB_TILES) == D.AB_ELEMS, (AB_TILES, D.AB_ELEMS)
AB_WIDE = D.AB_LANES > 32
HALF_OUTER = D.GLUE_HALF_OUTER
GLUE_FLAGS = {} if NHEAD == GLUE_NHEAD_DEFAULT else {"compile_flags": [f"-DDNGLUE_NHEAD={NHEAD}"]}

# ---- the full layer's attention cores (ax.py)
from recipes.attnknobs import probe_env  # noqa: E402
ATTN_FLAGS = [f"-DATTN_NH={NH}", f"-DATTN_KVH={KVH}", f"-DATTN_HD={HD}", f"-DATTN_ROT={A.ROT}", "-DATTN_GATE=1",
              f"-DATTN_VEXP={A.VEXP}", f"-DATTN_NHL={A.NHL}"]
if A.RB > 1:
    ATTN_FLAGS.append(f"-DATTN_RB={A.RB}")
for _k, _v in probe_env().items():
    if _k not in ("ATTN_RB", "ATTN_FAST"):
        ATTN_FLAGS.append(f"-D{_k}={_v}")
ACORES, NHL, RB = A.ACORES, A.NHL, A.RB
N_OG = NHL // A.HPO                              # og elements one attention core emits
if ACORES < 2 or N_OG != 1:
    sys.exit(f"dux.py: the og join assumes several attention cores of one og element each (ACORES={ACORES}, "
             f"N_OG={N_OG})")
OG_BYTES = NHL * HD * 2                          # one core's og element
# The attention cores walk the cached window BEFORE taking the new token's k and v (which only
# the new row and the cache write need), so the window streams while the main cores are still
# on gate | k | v. Same operations in the same order on the softmax state (cached rows, then the
# new row), so the same bytes out. DUX_WINDOW_FIRST=0 at export builds ax.py's order.
WINDOW_FIRST = os.environ.get("DUX_WINDOW_FIRST", "1") != "0"
OGJ_TILE = Tile(3, 1)                            # the memtile joining attention cores 1..5's og

PART = int(os.environ.get("DUX_PART", 0))        # 0 = the linear stream (lx), 1 = the full stream (ax)
if PART not in (0, 1):
    sys.exit(f"dux.py: DUX_PART={PART} (0 = lx, 1 = ax)")

# ONE set of buffer-argument shapes for both streams, so both builds emit the same image.
CONSTS_T = max(C_BYTES, CA_BYTES)
ACT_T = max(A_BYTES, AA_BYTES)
STATE_T = max(STATE_BYTES, KV_BYTES)
GLUE_TILE = Tile(2, 4)


def rows3(t: int):
    """lx.py's: tile t of each conv-state row, in bytes of the state BO (merged total)."""
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, STATE_T), t * TILE * 2, [1, 1, CONV_ROWS, TILE * 2], [0, 0, NCH * 2, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dux(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
        srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    u8_4k = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    u8_2k = np.ndarray[(2048,), np.dtype[np.uint8]]
    u8_ln = tl["u8_ln"]
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CONSTS_T,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_T,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(ACT_T,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]
    rtp_ty = np.ndarray[(3,), np.dtype[np.int32]]
    # glue / post (lx.py)
    nw_ty = np.ndarray[(SPEC.lin_value_dim,), np.dtype[bfloat16]]
    f32 = np.ndarray[(NHEAD,), np.dtype[np.float32]]
    facc = np.ndarray[(D.AB_LANES,), np.dtype[np.float32]] if AB_WIDE else f32
    fqk = np.ndarray[(2 * D.KEY_WIDTH,), np.dtype[np.float32]]
    fvt = np.ndarray[(TILE,), np.dtype[np.float32]]
    fxn = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    # attention (ax.py)
    u8_1k = np.ndarray[(A.E_A,), np.dtype[np.uint8]]
    b256 = np.ndarray[(HD,), np.dtype[bfloat16]]
    b512 = np.ndarray[(KVW,), np.dtype[bfloat16]]
    f64 = np.ndarray[(A.ROT,), np.dtype[np.float32]]
    f256 = np.ndarray[(HD,), np.dtype[np.float32]]
    fq = np.ndarray[(2 * QW,), np.dtype[bfloat16]] if A.VEXP else np.ndarray[(QW,), np.dtype[np.float32]]
    fml = np.ndarray[(2 * A.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    pb_ty = np.ndarray[(8 if RB > 1 else 4,), np.dtype[np.int32]]
    ogj_ty = np.ndarray[((ACORES - 1) * KVW,), np.dtype[bfloat16]]

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(ATTN), str(X.LN), str(X.LINL), str(X.RT),
                            str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    f_ab = (ExternalFunction("glue_ab_w", source_file=str(GLUE / "glue_ab_w.cc"),
                             arg_types=[u8_4k, fxn, facc, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
            if AB_WIDE else
            ExternalFunction("glue_ab_e", source_file=str(GLUE / "glue_ab_e.cc"),
                             arg_types=[u8_4k, fxn, f32, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS))
    f_small = ExternalFunction("glue_small_fn", source_file=str(GLUE / "glue_small.cc"),
                               arg_types=[u8_4k, facc, facc, f32, f32], include_dirs=inc, **GLUE_FLAGS)
    f_conv = ExternalFunction("glue_conv", source_file=str(GLUE / "glue_conv.cc"),
                              arg_types=[u8_2k, u8_2k, u8_2k, u8_2k, u8_2k, u8_4k, u8_4k, u8_2k, u8_2k, u8_2k, fqk, fvt,
                                         np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_emit = ExternalFunction("glue_emit_fn", source_file=str(GLUE / "glue_emit.cc"),
                              arg_types=[fqk, fvt, f32, f32, u8_2k, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_copy = ExternalFunction("glue_copy_xn_e", source_file=str(GLUE / "glue_copy_e.cc"),
                              arg_types=[u8_4k, fxn, np.int32], include_dirs=inc, **GLUE_FLAGS)
    post_fn = ExternalFunction("post_fn", source_file=str(POST / "post.cc"), arg_types=[u8_4k, u8_4k, nw_ty, u8_2k],
                               include_dirs=inc)
    post_copy = ExternalFunction("post_copy_nw", source_file=str(POST / "post_copy.cc"), arg_types=[u8_4k, nw_ty],
                                 include_dirs=inc)

    def af(sym, args):
        return ExternalFunction(sym, source_file=str(ATTN / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=ATTN_FLAGS)

    h0_arg = [np.int32]                                        # ACORES > 1 (asserted)
    f_meta = af("attn_meta", [u8_1k, u8_1k, b256, b256, f64, pb_ty])
    f_q = af("attn_q", [u8_1k, b256, f64, fq, np.int32])
    f_k = af("attn_k", [u8_1k, b256, f64, f256, b512, np.int32])
    f_v = af("attn_v", [u8_1k, b512, np.int32])
    f_init = af("attn_init", [foacc, fml])
    f_step = af("attn_step", [u8_1k, u8_1k, fq, foacc, fml, pb_ty] + h0_arg)
    f_stepn = af("attn_step_new", [b512, b512, fq, foacc, fml] + h0_arg)
    f_stepb = af("attn_stepb", [u8_1k] * (2 * RB) + [fq, foacc, fml, pb_ty] + h0_arg) if RB > 1 else None
    f_fin = af("attn_fin", [foacc, fml, u8_1k, u8_1k, b512, np.int32])

    # ---- fifos: the union of lx.py's and ax.py's, attention cores 1.. joined
    of_w = [ObjectFifo(t["elem"], name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(t["y"], name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(t["x"], name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=X.LNI_DEPTH)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_side = ObjectFifo(u8_4k, name="side", depth=2)
    of_gact = ObjectFifo(u8_2k, name="gact", depth=5)
    of_gout = ObjectFifo(u8_2k, name="gout", depth=3)
    of_pin = ObjectFifo(u8_4k, name="pin", depth=2)
    of_pout = ObjectFifo(u8_2k, name="pout", depth=2)
    of_ain = ObjectFifo(u8_1k, name="ain", depth=max(4, 2 * RB + 2))
    of_aout = ObjectFifo(b512, name="aout", depth=2)
    of_ogj = ObjectFifo(ogj_ty, name="ogj", depth=2)
    og_subs = of_ogj.prod().join([c * KVW for c in range(ACORES - 1)], tile=OGJ_TILE,   # offsets in ELEMENTS (bf16)
                                 depths=[2] * (ACORES - 1), obj_types=[b512] * (ACORES - 1),
                                 names=[f"og{c}" for c in range(1, ACORES)])

    # The per-core parameter words. Zeros in the image: the counts belong to the stream.
    rtp = [Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(3, dtype=np.int32), use_write_rtp=True)
           for c in range(N_CORES)]

    # ---- the main cores: lx's dense program with nA and nh read at run time
    def main_body(win, xin, yout, my_rtp, *args):
        B, K = X.unpack_args(args)
        tab = B["tab"]
        X.prep_stream(xin, K["prep"], tab, HID, XN_ELEMS)      # the xn: the stream wrote the words first
        nbands = my_rtp[0]                                     # QKV_PC + Z_PC | 2 Q_PC + 2 KV_PC
        nheads = my_rtp[1]                                     # DN_HEADS_PC | 0
        nslices = my_rtp[2]                                    # DN_SLICES (run-time: no unrolled pass 1)
        X.gemv_bands(win, yout, tab, K["gy"], nbands, X.n_groups(HID), X.per_band(HID), 2)
        X.dn_body(win, yout, B, K, nheads, nslices)
        X.prep_bands(win, xin, yout, B, K, OUT_K, OG_ELEMS, OUT_PC, "linear_out")
        X.ffn_body(win, xin, yout, B, K)

    # ---- glue and post (lx.py's dense bodies)
    def glue_body(sin, ain, oout, acc_a, acc_b, decay, beta, qk, vt, xn, fab, fsmall, fconv, femit, fcopy):
        if HALF_OUTER:
            for h, ntiles in enumerate(AB_TILES):
                e0 = sin.acquire(1)
                fcopy(e0, xn, 0)
                sin.release(1)
                for acc in (acc_a, acc_b):
                    for tile in range_(ntiles):
                        ww = sin.acquire(1)
                        fab(ww, xn, acc, tile, 1 if h == 0 else 0)
                        sin.release(1)
        else:
            for acc in (acc_a, acc_b):
                for h, ntiles in enumerate(AB_TILES):
                    e0 = sin.acquire(1)
                    fcopy(e0, xn, 0)
                    sin.release(1)
                    for tile in range_(ntiles):
                        ww = sin.acquire(1)
                        fab(ww, xn, acc, tile, 1 if h == 0 else 0)
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

    # ---- the attention cores (ax.py's _attn)
    def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
              fm, fq_, fk, fv, fi, fs, fsn, ff, fsb, c):
        h0 = c * NHL
        e = ain.acquire(2)
        fm(e[0], e[1], qn, kn, cs, pb)
        ain.release(2)
        for h in range_(A.Q_AIN_ELEMS):
            e = ain.acquire(1)
            fq_(e, qn, cs, qs, h)
            ain.release(1)

        def new_kv():
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

        def window():
            fi(oacc, ml)
            if RB > 1:
                for _ in range_(pb[4]):
                    e = ain.acquire(2 * RB)
                    fsb(*([e[i] for i in range(2 * RB)] + [qs, oacc, ml, pb, h0]))
                    ain.release(2 * RB)
                for _ in range_(pb[5]):
                    e = ain.acquire(2)
                    fs(e[0], e[1], qs, oacc, ml, pb, h0)
                    ain.release(2)
            else:
                for _ in range_(pb[1]):
                    e = ain.acquire(2)
                    fs(e[0], e[1], qs, oacc, ml, pb, h0)
                    ain.release(2)

        if WINDOW_FIRST:
            window()
            new_kv()
        else:
            new_kv()
            window()
        fsn(kout, vout, qs, oacc, ml, h0)
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

    if RB > 1:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq_, fk, fv, fi, fs, fsn, ff, fsb):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq_, fk, fv, fi, fs, fsn, ff, fsb, 0)

        def make_attn_body(c):
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq_, fk, fv, fi, fs, fsn, ff, fsb):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq_, fk, fv, fi, fs, fsn, ff,
                      fsb, c)
            return body
    else:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq_, fk, fv, fi, fs, fsn, ff):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq_, fk, fv, fi, fs, fsn, ff, None, 0)

        def make_attn_body(c):
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq_, fk, fv, fi, fs, fsn, ff):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fq_, fk, fv, fi, fs, fsn, ff,
                      None, c)
            return body

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(b256, name=f"qn{s}"), Buffer(b256, name=f"kn{s}"), Buffer(f64, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(f256, name=f"tmp{s}"), Buffer(b512, name=f"kout{s}"),
                Buffer(b512, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]

    ln_fn, ln_args = X.ln_dense_worker(of_lni, of_lno, L)
    workers = [Worker(ln_fn, fn_args=ln_args, tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body,
                              fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), rtp[c],
                                       *X.worker_args(X.core_buffers(t, c), K)],
                              tile=Tile(c, 2), stack_size=0x1800))
    workers.append(Worker(post_body, fn_args=[of_pin.cons(), of_pout.prod(), Buffer(nw_ty, name="nwb"), post_fn, post_copy],
                          tile=Tile(1, 3), stack_size=0x1800))
    workers.append(Worker(glue_body,
                          fn_args=[of_side.cons(), of_gact.cons(), of_gout.prod(),
                                   Buffer(facc, name="acc_a"), Buffer(facc, name="acc_b"), Buffer(f32, name="decay"),
                                   Buffer(f32, name="beta"), Buffer(fqk, name="qk"), Buffer(fvt, name="vt"),
                                   Buffer(fxn, name="xnb"), f_ab, f_small, f_conv, f_emit, f_copy],
                          tile=GLUE_TILE, stack_size=0x1800))
    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin] + ([f_stepb] if RB > 1 else [])
    workers.append(Worker(attn_body, fn_args=[of_ain.cons(), of_aout.prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=0x1800))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn_body(c), fn_args=[of_ain.cons(), og_subs[c - 1].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=0x1800))

    bt = X.bt
    YB = X.BAND_ROWS * 4

    def set_rtp(nbands: int, nheads: int):
        for c in range(N_CORES):
            rtp[c][0] = nbands
            rtp[c][1] = nheads
            rtp[c][2] = X.DN_SLICES
        if X.STAMP:
            # LX_STAMP timing builds read a tile timer over the processor bus (gen_kernels.py);
            # without Core_Processor_Bus.Enable the core hangs on its first lda.tm.
            from aie.dialects.aiex import npu_maskwrite32
            for c in range(N_CORES):
                npu_maskwrite32(column=c, row=2, address=0x32038, value=0x1, mask=0x1)

    def dn_post_sequence(pw, py, a_state, a_act, a_consts, w_prods, y_conss, pin_p, pout_c):
        """DeltaNet on the main cores with post overlapped. xcommon.dn_sequence gives core c heads
        c*6 .. c*6+5, so post's group g (heads 8g .. 8g+7) needs two cores' LAST heads and waits
        for all of DeltaNet. Here round h is heads 8h .. 8h+7, one per core -- post's group h
        exactly -- so post runs group h while the cores run head h+1 and only the last group is
        left after the last head. A head is self-contained (its record carries k, q, v, decay and
        beta; S, the record and o sit at the head's own offsets), so which core runs it changes
        no byte. Each endpoint keeps xcommon's issue order (head-major, the #113 lesson).
        Queue state that makes the waits exact: after round h is issued, every y queue holds
        [o(h-1), S'(h), o(h)] (depth 3), so finish_oldest is exactly round h-1's o drains."""
        assert NG == DN_HEADS_PC and G == N_CORES * SPEC.lin_value_dim, (NG, DN_HEADS_PC, G)
        rec, ohb = D.RECORD_BYTES, D.O_HEAD_BYTES
        pp = Pipeline(3)
        pp.fill(pin_p, a_consts, bt(CONSTS_T, C_NW, ELEM))

        def post_group(g):
            pp.drain(pout_c, a_act, bt(ACT_T, A_OG + g * G * 2, G * 2))
            pp.fill(pin_p, a_act, bt(ACT_T, A_O + g * G * 4, G * 4))
            pp.fill(pin_p, a_act, bt(ACT_T, A_Z + g * G * 4, G * 4))

        def s_read(off: int):
            """One head's S as DN_SLICES w elements -- xcommon.dn_sequence's s_tap against the merged
            total: a plain linear fill where a slice is the whole element; t2's 2176 B element
            carries 4 rows (2048 B), so the slices are read at that stride, each element running
            128 B into the next slice (never read; the state buffer has a tail for the last one)."""
            if X.DN_SLICE == X.CALL_BYTES:
                return bt(STATE_T, off, S_HEAD_BYTES)
            assert off + (X.DN_SLICES - 1) * X.DN_SLICE + X.CALL_BYTES <= STATE_BYTES, (off, STATE_BYTES)
            return TensorAccessPattern((1, STATE_T), off, [1, 1, X.DN_SLICES, X.CALL_BYTES], [0, 0, X.DN_SLICE, 1])

        for h in range(DN_HEADS_PC):
            for c in range(N_CORES):
                hd = h * N_CORES + c
                s_off = STATE_S_OFF + hd * S_HEAD_BYTES
                pw.fill(w_prods[c], a_act, bt(ACT_T, A_VEC + hd * rec, X.CALL_BYTES))
                pw.fill(w_prods[c], a_state, s_read(s_off))
                py.drain(y_conss[c], a_state, bt(STATE_T, s_off, S_HEAD_BYTES))     # S' back, whole rows
                py.drain(y_conss[c], a_act, bt(ACT_T, A_O + hd * ohb, ohb))
                pw.fill(w_prods[c], a_state, s_read(s_off))
            if h:
                assert all(len(py._q(e)) == 3 for e in y_conss)
                py.finish_oldest(*y_conss)                 # round h-1's o: post group h-1 is in DDR
                post_group(h - 1)
        py.finish(*y_conss)                                # the last round's o
        post_group(DN_HEADS_PC - 1)
        pp.finish()                                        # og is in DDR

    # ---- part 0: lx.py's dense_sequence, against the merged totals
    def linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                        side_p, gact_p, gout_c, pin_p, pout_c):
        set_rtp(NBANDS_LIN, DN_HEADS_PC)
        BB_HID, BB_OUT = X.role_band_bytes("linear", HID), X.role_band_bytes("linear_out", OUT_K)
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(CONSTS_T, C_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(ACT_T, A_XN, ELN), wait=True, group=tg_ln)
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
            py.drain(y_conss[c], a_act, bt(ACT_T, A_QKV + c * QKV_PC * YB, QKV_PC * YB))
            py.drain(y_conss[c], a_act, bt(ACT_T, A_Z + c * Z_PC * YB, Z_PC * YB))
        tg_ln.finish()
        px.fill(x_prod, a_act, bt(ACT_T, A_XN, XN_ELEMS * ELEM))
        ps = Pipeline(3)
        if HALF_OUTER:
            off = 0
            for h, ntiles in enumerate(AB_TILES):
                ps.fill(side_p, a_act, bt(ACT_T, A_XN + h * ELEM, ELEM))
                for reg in (SIDE_ALPHA, SIDE_BETA):
                    ps.fill(side_p, a_consts, bt(CONSTS_T, C_SIDE + reg + off, ntiles * ELEM))
                off += ntiles * ELEM
        else:
            for reg in (SIDE_ALPHA, SIDE_BETA):
                off = 0
                for h, ntiles in enumerate(AB_TILES):
                    ps.fill(side_p, a_act, bt(ACT_T, A_XN + h * ELEM, ELEM))
                    ps.fill(side_p, a_consts, bt(CONSTS_T, C_SIDE + reg + off, ntiles * ELEM))
                    off += ntiles * ELEM
        ps.fill(side_p, a_consts, bt(CONSTS_T, C_SIDE + SIDE_SMALL, ELEM))
        ps.fill(side_p, a_consts, bt(CONSTS_T, C_SIDE + SIDE_CONV, GLUE_SIDE_BYTES - SIDE_CONV))
        # qkv is in DDR. Only qkv (lx.py's dense stream waits for z too): the glue reads A_QKV
        # and nothing else, and every core computes its QKV_PC qkv bands before its Z_PC z bands,
        # so the glue's conv and records now run while the cores are still on z. z is first read
        # by post, behind the py.finish() after DeltaNet. The MoE stream has done this since #113.
        py.finish_oldest(*y_conss)
        pipe = Pipeline(3)
        for tt in range(NT):
            pipe.drain(gout_c, a_state, rows3(tt))
            if tt >= KEY_TILES:
                pipe.drain(gout_c, a_act, bt(ACT_T, A_VEC + (tt - KEY_TILES) * D.HEADS_PER_TILE * D.RECORD_BYTES,
                                             D.HEADS_PER_TILE * D.RECORD_BYTES))
            pipe.fill(gact_p, a_act, bt(ACT_T, A_QKV + tt * TILE * 4, TILE * 4))
            pipe.fill(gact_p, a_state, rows3(tt))
        pipe.finish()
        ps.finish()
        dn_post_sequence(pw, py, a_state, a_act, a_consts, w_prods, y_conss, pin_p, pout_c)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_consts, bt(CONSTS_T, C_WOUT + c * OUT_PC * BB_OUT, OUT_PC * BB_OUT))
            py.drain(y_conss[c], a_act, bt(ACT_T, A_OUT + c * OUT_PC * YB, OUT_PC * YB))
        px.fill(x_prod, a_act, bt(ACT_T, A_OG, OG_ELEMS * ELEM))
        py.finish()
        X.ln_split_residual_norm(lni, lno, c_xres, a_act, a_consts, ACT_T, CONSTS_T, A_OUT, A_RES, A_XM, C_POSTLN)
        X.ffn_sequence(pw, px, py, a_pool, a_act, w_prods, x_prod, y_conss,
                       ACT_T, A_XM, A_H, A_OUT2, POOL_FFN_UP, POOL_FFN_GATE, POOL_FFN_DOWN, A_OUT2B)
        py.finish()
        X.ln_split_close(lni, lno, c_xres, a_act, ACT_T, A_RES, A_OUT2, A_OUT2B)
        pw.finish()
        px.finish()

    # ---- part 1: ax.py's dense_sequence, against the merged totals, og through the join
    def full_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                      ain_p, aout_c, ogj_c):
        set_rtp(NBANDS_FULL, 0)
        BB_HID, BB_O = X.role_band_bytes("attn", HID), X.role_band_bytes("attn", O_K)

        def w_regions(c):
            return [(POOL_Q + c * Q_PC * BB_HID, Q_PC * BB_HID), (POOL_GATE + c * Q_PC * BB_HID, Q_PC * BB_HID),
                    (POOL_K + c * KV_PC * BB_HID, KV_PC * BB_HID), (POOL_V + c * KV_PC * BB_HID, KV_PC * BB_HID),
                    (POOL_O + c * O_PC * BB_O, O_PC * BB_O)]

        def y_regions(c):
            qb, kb = Q_PC * YB, KV_PC * YB
            return [(AA_QG + c * qb, qb), (AA_QG + QW * 4 + c * qb, qb),
                    (AA_KVN + c * kb, kb), (AA_KVN + KVW * 4 + c * kb, kb),
                    (AA_OUT + c * O_PC * YB, O_PC * YB)]

        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(CONSTS_T, CA_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(ACT_T, AA_XN, ELN), wait=True, group=tg_ln)
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            for off, n in w_regions(c)[:3]:
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, off, n))
            for off, n in y_regions(c)[:3]:
                py.drain(y_conss[c], a_act, bt(ACT_T, off, n))
        tg_ln.finish()
        px.fill(x_prod, a_act, bt(ACT_T, AA_XN, XN_ELEMS * ELEM))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[3]))
            py.drain(y_conss[c], a_act, bt(ACT_T, *y_regions(c)[3]))
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        pa_out.drain(aout_c, a_kv, bt(STATE_T, KV_ROW, KV_ROW))           # the new row [k' | v'] (attnpos)
        pa_out.drain(aout_c, a_act, bt(ACT_T, AA_OG, OG_BYTES))           # attention core 0's og
        pa_out.drain(ogj_c, a_act, bt(ACT_T, AA_OG + OG_BYTES, (ACORES - 1) * OG_BYTES))   # cores 1..: joined
        pa_in.fill(ain_p, a_consts, bt(CONSTS_T, CA_META, A.E_A))         # [qn | kn]
        pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, PTAB_ROW, PTAB_ROW))     # the position record (attnpos)
        # q is in DDR already: issuing each core's 4th drain (v) made the throttle await its
        # oldest, the q drain. The attention cores take q first, so q goes now -- their q stage
        # runs while the main cores are still on gate | k | v (the MoE stream's order, ax.py).
        assert all(len(py._q(e)) == 3 for e in y_conss), "q's drain must be the one retired"
        pa_in.fill(ain_p, a_act, bt(ACT_T, AA_QG, QW * 4))
        if WINDOW_FIRST:
            pa_in.fill(ain_p, a_kv, bt(STATE_T, 0, KV_ROW))               # the window: rows [0, nf) (attnpos)
        py.finish(*y_conss)                                               # gate, k, v are in DDR
        # The o projection's weights and drains next: every q/gate/k/v transfer has landed, so the
        # throttle's waits are already met, and the w fifos fill while attention runs instead of
        # behind the gate fill (which, window first, waits for the window to be taken).
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[4]))
            py.drain(y_conss[c], a_act, bt(ACT_T, *y_regions(c)[4]))
        pa_in.fill(ain_p, a_act, bt(ACT_T, AA_KVN, KVW * 4))
        pa_in.fill(ain_p, a_act, bt(ACT_T, AA_KVN + KVW * 4, KVW * 4))
        if not WINDOW_FIRST:
            pa_in.fill(ain_p, a_kv, bt(STATE_T, 0, KV_ROW))               # the window: rows [0, nf) (attnpos)
        pa_in.fill(ain_p, a_act, bt(ACT_T, AA_QG + QW * 4, QW * 4))
        pa_out.finish()                                                   # og (and the new cache row) are in DDR
        px.fill(x_prod, a_act, bt(ACT_T, AA_OG, OG_ELEMS * ELEM))
        py.finish()
        X.ln_split_residual_norm(lni, lno, c_xres, a_act, a_consts, ACT_T, CONSTS_T, AA_OUT, AA_RES, AA_XM, CA_POSTLN)
        X.ffn_sequence(pw, px, py, a_pool, a_act, w_prods, x_prod, y_conss,
                       ACT_T, AA_XM, AA_H, AA_OUT2, POOL_FFN_UP, POOL_FFN_GATE, POOL_FFN_DOWN, AA_OUT2B)
        py.finish()
        X.ln_split_close(lni, lno, c_xres, a_act, ACT_T, AA_RES, AA_OUT2, AA_OUT2B)
        pw.finish()
        px.finish()
        pa_in.finish()

    def sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                 side_p, gact_p, gout_c, pin_p, pout_c, ain_p, aout_c, ogj_c):
        if part == 0:
            linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                            side_p, gact_p, gout_c, pin_p, pout_c)
        else:
            full_sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                          ain_p, aout_c, ogj_c)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)), of_gout.cons(tile=Tile(2, 0)),
                            of_pin.prod(tile=Tile(4, 0)), of_pout.cons(tile=Tile(1, 0)),
                            of_ain.prod(tile=Tile(5, 0)), of_aout.cons(tile=Tile(4, 0)), of_ogj.cons(tile=Tile(3, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = dux
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes(), (HERE / "dux.py").read_bytes()] + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.cc"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.h")) + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(),
                   (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(), (GEMV / "wht.h").read_bytes(),
                   (GEMV / "gemv_t2.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode(), b"window_first=%d" % WINDOW_FIRST])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
