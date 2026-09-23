r"""ax: a whole full-attention layer (attention block + MoE block) in ONE xclbin
context (phase 2 "whole-layer context"; the linear-layer twin is lx.py):

    ln -> gemv q | gate | k | v -> attn -> gemv o -> ln (+residual) -> router -> MoE

Same recipe as lx: 8 main cores (Tile(c, 2)) with the w / x / y streams run
every GEMV then the MoE; the ln + router core (Tile(0, 3)) and the attention
core (Tile(2, 3), attn.py's verbatim) are the helpers. Two instruction streams
on one xclbin (CompileTime `part`): 0 = everything up to the router, 1 = the
MoE (after the driver's `moeroute2`).

The cache position is NOT a build parameter (plan item 3, dynamic KV length):
the stream is built for the placeholder position 1 -- the KV window is ONE
linear fill of rows [0, nf) (the fifo delivers it as 2 nf 1 KB elements, K_t
V_t in order), the new row ONE 2 KB drain to row pos, the position record
(pos, nf, RoPE cos/sin) one 1 KB fill from ptab row pos -- and the driver's
`attnpos ax0 <pos>` rewrites those three words per token (the fill's BD
length, the two offsets). The attention core loops over the record's nf =
max(pos, 1) rows and masks rows t >= pos (position 0 streams one dummy row).

Geometry: the recipe's (open_kernels/recipes/qwen36moe.py, `R.attn`) for the
spec named by OPEN_KERNELS_SPEC (else the checked-in 27B); attn.h's head
count / dims are the compile-time macros ATTN_NH / ATTN_KVH / ATTN_HD /
ATTN_ROT set from the same spec.

Args (layout.py): pool (q/k/v/gate/o and the experts at their pool offsets),
xres f32[HID] (in: layer input; out: layer output), consts [lnw | postln |
meta (qn | kn) | router W | sgw], kv (the layer's KV cache: MAX_CTX rows of
[K_t | V_t], updated in place), act (scratch), ptab (the position record
table, shared by the attention layers).
Build (WSL): for p in 0 1: AX_PART=$p python build_design.py designs/layer_x/ax.py designs/layer_x/build_ax$p
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
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (AA_BYTES, AA_H, AA_HP, AA_KVN, AA_OG, AA_OUT, AA_OUT2, AA_QG, AA_RES, AA_ROUT,  # noqa: E402
                    AA_XM, AA_XN, CA_BYTES, CA_LNW, CA_META, CA_POSTLN, CA_RW, CA_SGW, ELN, KV_BYTES,
                    KV_ROW, POOL_BYTES, POOL_FFN_DOWN, POOL_FFN_GATE, POOL_FFN_UP, POOL_GATE, POOL_K,
                    POOL_O, POOL_Q, POOL_V, PTAB_BYTES, PTAB_ROW, R, SPEC)
import xcommon as X  # noqa: E402
import xlayer as XL  # noqa: E402

D = R.attn
if D is None:
    sys.exit("ax.py: the spec has no full-attention layers")
HID = X.HID
NH, KVH, HD = D.NH, D.KVH, D.HD
N_CORES = X.N_CORES
ELEM = X.ELEM
Q_PC, KV_PC, O_PC = D.Q_PC, D.KV_PC, D.O_PC             # bands per core: q (and gate), k (and v), o
QW, KVW, O_K = D.QW, D.KVW, D.O_K
DENSE = X.KIND == "dense"                               # the Qwen3.5 composition: a dense FFN tail
PART = int(os.environ.get("AX_PART", 0))
if DENSE and PART:
    sys.exit("ax.py: the dense tail is one instruction stream; AX_PART must be 0")
XN_ELEMS = X.FFN.XN_ELEMS if DENSE else 1               # 4 KB x elements the xn arrives in
OG_ELEMS = D.OG_ELEMS
# The attention cores (their flags, kernels, bodies, fifos) and this layer's part-0 host
# sequence live in xlayer.py, shared with the merged image (ux.py).
ACORES, NHL = XL.ACORES, XL.NHL
w_regions, y_regions = XL.w_regions, XL.y_regions


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ax(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
       srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    a = XL.attn_types()
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CA_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(KV_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(AA_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]

    inc = include_dirs() + [str(GEMV), str(ATTN), str(X.LN), str(X.LINL), str(X.RT), str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    afns = XL.attn_kernels(inc, a)

    # ---- fifos
    of_w, of_y, of_x = XL.main_fifos(t)
    of_lni, of_lno = XL.ln_fifos(tl)
    of_ain, of_aout, of_og = XL.attn_fifos(a)

    def main_body(win, xin, yout, *args):
        B, K = X.unpack_args(args)
        tab = B["tab"]
        if DENSE:
            # ONE stream: q | gate | k | v, the o projection, then the dense FFN tail.
            X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS, 2 * Q_PC + 2 * KV_PC, "attn")
            X.prep_bands(win, xin, yout, B, K, O_K, OG_ELEMS, O_PC, "attn")
            X.ffn_body(win, xin, yout, B, K)
            return
        xe = xin.acquire(1)                                     # xn
        K["prep2048"](xe, tab)
        X.role_gemv_bands(win, yout, B, K, "attn", 2 * Q_PC + 2 * KV_PC, HID)   # q | gate | k | v
        xin.release(1)
        oe = xin.acquire(2)                                     # og, K = QW
        K["prep4096a"](oe[0], tab)
        K["prep4096b"](oe[1], tab)
        X.role_gemv_bands(win, yout, B, K, "attn", O_PC, O_K)
        xin.release(2)
        X.moe_body(win, xin, yout, B, K)

    workers = ([XL.ln_worker(of_lni, of_lno, tl, L)] + XL.main_workers(main_body, of_w, of_x, of_y, t, K)
               + XL.attn_workers(of_ain, of_aout, of_og, a, afns))

    bt = X.bt

    def dense_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                       ain_p, aout_c, og_cs):
        """ONE instruction stream: the MoE stream's attention half with the router dropped,
        then designs/dense/dx.py's steps 5-7 (residual + norm, the FFN, the output residual)."""
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(CA_BYTES, CA_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(AA_BYTES, AA_XN, ELN), wait=True, group=tg_ln)
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            for off, n in w_regions(c)[:3]:
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, off, n))
            for off, n in y_regions(c)[:3]:
                py.drain(y_conss[c], a_act, bt(AA_BYTES, off, n))
        tg_ln.finish()                                            # xn is in DDR
        px.fill(x_prod, a_act, bt(AA_BYTES, AA_XN, XN_ELEMS * ELEM))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[3]))
            py.drain(y_conss[c], a_act, bt(AA_BYTES, *y_regions(c)[3]))
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        pa_out.drain(aout_c, a_kv, bt(KV_BYTES, KV_ROW, KV_ROW))        # the new row [k' | v'] (attnpos)
        pa_out.drain(aout_c, a_act, bt(AA_BYTES, AA_OG, NHL * HD * 2))
        for c in range(1, ACORES):                                      # heads NHL*c ..
            pa_out.drain(og_cs[c - 1], a_act, bt(AA_BYTES, AA_OG + c * NHL * HD * 2, NHL * HD * 2))
        pa_in.fill(ain_p, a_consts, bt(CA_BYTES, CA_META, D.E_A))       # [qn | kn]
        pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, PTAB_ROW, PTAB_ROW))   # the position record (attnpos)
        py.finish(*y_conss)                                       # q, gate, k, v are in DDR
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_QG, QW * 4))
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_KVN, KVW * 4))
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_KVN + KVW * 4, KVW * 4))
        pa_in.fill(ain_p, a_kv, bt(KV_BYTES, 0, KV_ROW))                # the window: rows [0, nf) (attnpos)
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_QG + QW * 4, QW * 4))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[4]))
            py.drain(y_conss[c], a_act, bt(AA_BYTES, *y_regions(c)[4]))
        pa_out.finish()                                           # og (and the new cache rows) are in DDR
        px.fill(x_prod, a_act, bt(AA_BYTES, AA_OG, OG_ELEMS * ELEM))
        py.finish()                                               # out is in DDR
        # res = xres + out; xm = post_attention_norm(res)
        tg_ln2 = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(CA_BYTES, CA_POSTLN, ELN), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(AA_BYTES, AA_RES, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(AA_BYTES, AA_XM, ELN), wait=True, group=tg_ln2)
        lni.fill(a_act, tap=bt(AA_BYTES, AA_OUT, HID * 4), wait=True, group=tg_ln2)
        tg_ln2.finish()                                           # res, xm are in DDR
        X.ffn_sequence(pw, px, py, a_pool, a_act, w_prods, x_prod, y_conss,
                       AA_BYTES, AA_XM, AA_H, AA_OUT2, POOL_FFN_UP, POOL_FFN_GATE, POOL_FFN_DOWN)
        tg_ln3 = TaskGroup()
        lni.fill(a_act, tap=bt(AA_BYTES, AA_RES, HID * 4), wait=True, group=tg_ln3)
        lni.fill(a_consts, tap=bt(CA_BYTES, CA_POSTLN, ELN), wait=True, group=tg_ln3)    # unused w
        lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln3)
        lno.drain(a_act, tap=bt(AA_BYTES, AA_XN, ELN), wait=True, group=tg_ln3)   # the junk xn, over spent AA_XN
        py.finish()                                               # out2 is in DDR
        lni.fill(a_act, tap=bt(AA_BYTES, AA_OUT2, HID * 4), wait=True, group=tg_ln3)
        tg_ln3.finish()
        pw.finish()
        px.finish()
        pa_in.finish()

    def sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c, og_cs):
        if DENSE:
            dense_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                           ain_p, aout_c, og_cs)
        elif part == 0:
            XL.full_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                             ain_p, aout_c, og_cs)
        else:
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                           AA_BYTES, CA_BYTES, AA_XM, AA_ROUT, AA_RES, AA_HP, CA_SGW)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0)),
                            [of_og[c].cons(tile=Tile(3 + c, 0)) for c in range(ACORES - 1)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = ax
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + [(HERE / "xcommon.py").read_bytes(), (HERE / "xlayer.py").read_bytes()]
                + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
