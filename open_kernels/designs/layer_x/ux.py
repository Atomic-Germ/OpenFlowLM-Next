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

NOTHING HERE IS COPIED
The glue / post helpers, the attention cores, the norm helper, the fifos and both
part-0 host sequences are xlayer.py's -- the same functions lx.py and ax.py call --
so an edit to either layer type reaches the merged image by construction. (This
file used to carry hand copies of them; two edits had to be ported by hand before
they were factored out.) What is this file's own: the main-core program with its
two RTP words, the glue's tile, the merged buffer totals, the RTP writes at the
head of parts 0 and 2, and the shim endpoints that move (`ain`, `aout`).

Build: for p in 0 1 2 3: UX_PART=$p python build_design.py designs/layer_x/ux.py designs/layer_x/build_ux$p
or, with the manifest, python export_qwen36_kernels.py --only lx0,lx1,ax0,ax1 --force (the default for
the qwen36moe family; OPEN_LAYER_ONE_CTX=0 builds lx.py / ax.py instead)
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, Program, Runtime
from aie.iron.device import Tile

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_HP, A_RES, A_ROUT, A_XM, AA_BYTES, AA_HP, AA_RES, AA_ROUT, AA_XM,  # noqa: E402
                    C_BYTES, C_SGW, CA_BYTES, CA_SGW, KV_BYTES, POOL_BYTES, PTAB_BYTES, STATE_BYTES, R, SPEC)
import xcommon as X  # noqa: E402
import xlayer as XL  # noqa: E402

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
QKV_PC, Z_PC, OUT_PC, OUT_K = D.QKV_PC, D.Z_PC, D.OUT_PC, D.OUT_K
Q_PC, KV_PC, O_PC, O_K = A.Q_PC, A.KV_PC, A.O_PC, A.O_K
if (OUT_PC, OUT_K, D.OG_ELEMS) != (O_PC, O_K, A.OG_ELEMS):
    sys.exit(f"ux.py: the two og projections differ -- linear {OUT_PC} bands of K={OUT_K} in "
             f"{D.OG_ELEMS} elements against full {O_PC} of K={O_K} in {A.OG_ELEMS}. One image "
             "can only share that stage while they match.")
OG_ELEMS = D.OG_ELEMS
NBANDS_LIN = QKV_PC + Z_PC                       # RTP word 0, linear stream
NBANDS_FULL = 2 * Q_PC + 2 * KV_PC               # RTP word 0, full stream
DN_HEADS_PC = X.DN_HEADS_PC                      # RTP word 1, linear stream (full: 0)

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


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ux(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
       srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    g = XL.glue_post_types()
    a = XL.attn_types()
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CONSTS_T,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_T,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(ACT_T,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(ATTN), str(X.LN), str(X.LINL), str(X.RT),
                            str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    GK = XL.glue_post_kernels(inc, g)
    afns = XL.attn_kernels(inc, a)

    # ---- fifos: the union of lx's and ax's
    of_w, of_y, of_x = XL.main_fifos(t)
    of_lni, of_lno = XL.ln_fifos(tl)
    of_side, of_gact, of_gout, of_pin, of_pout = XL.glue_post_fifos(g)
    of_ain, of_aout, of_og = XL.attn_fifos(a)

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
        X.dn_body(win, yout, B, K, nheads)
        oe = xin.acquire(2)                                 # og, K = OUT_K (= O_K)
        K["prep4096a"](oe[0], tab)
        K["prep4096b"](oe[1], tab)
        X.role_gemv_bands(win, yout, B, K, "linear_out", OUT_PC, OUT_K)
        xin.release(2)
        X.moe_body(win, xin, yout, B, K)

    workers = ([XL.ln_worker(of_lni, of_lno, tl, L)]
               + XL.main_workers(main_body, of_w, of_x, of_y, t, K, extra=lambda c: [rtp[c]])
               + XL.glue_post_workers(of_side, of_gact, of_gout, of_pin, of_pout, g, GK, GLUE_TILE)
               + XL.attn_workers(of_ain, of_aout, of_og, a, afns))

    # ---- host sequences (one per instruction stream)
    def linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                        side_p, gact_p, gout_c, pin_p, pout_c):
        """lx.py's part 0 (xlayer.linear_sequence), against the merged buffer totals, after
        the two RTP words."""
        for c in range(N_CORES):
            rtp[c][0] = NBANDS_LIN
            rtp[c][1] = DN_HEADS_PC
        XL.linear_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                           side_p, gact_p, gout_c, pin_p, pout_c, act_t=ACT_T, consts_t=CONSTS_T, state_t=STATE_T)

    def full_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                      ain_p, aout_c, og_cs):
        """ax.py's part 0 (xlayer.full_sequence), against the merged buffer totals, after
        the two RTP words."""
        for c in range(N_CORES):
            rtp[c][0] = NBANDS_FULL
            rtp[c][1] = 0
        XL.full_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                         ain_p, aout_c, og_cs, act_t=ACT_T, consts_t=CONSTS_T, kv_t=STATE_T)

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
                            [of_og[c].cons(tile=Tile(3 + c, 0)) for c in range(XL.ACORES - 1)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = ux
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes(), (HERE / "xlayer.py").read_bytes(), (HERE / "ux.py").read_bytes()] + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.cc"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.h")) + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(),
                   (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), (HERE.parent.parent / "include" / "scalar_fp.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
