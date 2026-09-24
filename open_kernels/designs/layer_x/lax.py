r"""lax: BOTH whole-layer types of the hybrid 35B in ONE xclbin context.

The 35B (`qwen36-35b-a3b`) is Qwen3-Next's 3:1 hybrid: 30 linear-attention layers
(lx.py's shape) and 10 full-attention layers (ax.py's shape). An `xrt::runlist` is
bound to ONE hw_context (= one xclbin UUID), so a per-token runlist of all 40 layers
needs both types in one xclbin. The tile programs (not just the control text) live in
the xclbin, and the main core has 112 B of program memory free, so the two layer types
cannot be concatenated into one body and cannot afford a runtime drain/dispatch branch
(see LAX-MERGE-NOTES.md).

They do not have to be. On the 35B *every* projection is q4_1 (`R.q8` is empty), so
the `role` argument of the GEMV is inert -- `linear`, `linear_out` and `attn` resolve
to the same kernel and the same group/band law. The two main bodies then differ in
exactly two places:

    lx: role_gemv_bands(..., QKV_PC + Z_PC = 24, HID); dn_body(...); (out, 4)
    ax: role_gemv_bands(..., 2*Q_PC + 2*KV_PC = 18, HID);              (out, 4)

So ONE main body -- lx.py's, byte for byte -- serves both layer types; only the *host
sequence* differs. For a full-attention layer the sequence simply fills 24 pre-MoE
bands (18 real q|gate|k|v + 6 dummy) and, because the body always runs the DeltaNet
step, feeds and sinks that step inside `state`, which a full-attention layer does not
use. The attention itself (the 4 attn cores at Tile(3+c, hrow)) and the glue/post
helpers are both present in the xclbin and only one of them is driven per dispatch.

The CompileTime `kind` (0 = linear, 1 = full) selects the host sequence. The xclbin is
identical for both builds -- same workers, same fifos, same tiles -- so the two
`insts.elf` streams share it and a runlist can hold both. The tile/channel budget is
the proof that the merge is real: 14 shim MM2S fills and 15 S2MM drains (lx is 13/11,
ax 14/11), and the two main-program sizes are unchanged at 16272 (lx) / 12064 (ax).

Args (8 buffers, the runtime's limit): pool, xres, consts, kv, act, ptab, state, cfg.
`kv`/`ptab` are dummies for a linear layer and `state` is a dummy for a full one.
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
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (AA_BYTES, AA_H, AA_HP, AA_KVN, AA_OG, AA_OUT, AA_OUT2, AA_QG, AA_RES, AA_ROUT,  # noqa: E402
                    AA_XM, AA_XN, A_BYTES, A_H, A_HP, A_O, A_OG, A_OUT, A_OUT2, A_QKV, A_RES, A_ROUT, A_VEC,
                    A_XM, A_XN, A_Z, C_BYTES, C_LNW, C_NW, C_POSTLN, C_RW, C_SGW, C_SIDE, C_WOUT,
                    CA_BYTES, CA_LNW, CA_META, CA_POSTLN, CA_RW, CA_SGW, ELN, GLUE_SIDE_BYTES, KV_BYTES,
                    KV_ROW, POOL_BYTES, POOL_FFN_DOWN, POOL_FFN_GATE, POOL_FFN_UP, POOL_GATE, POOL_K,
                    POOL_O, POOL_Q, POOL_QKV, POOL_V, POOL_Z, PTAB_BYTES, PTAB_ROW, SIDE_ALPHA,
                    SIDE_BETA, SIDE_CONV, SIDE_SMALL, STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES, R, SPEC)
import xcommon as X  # noqa: E402

D = R.linear
DA = R.attn
if D is None or DA is None:
    sys.exit("lax.py: the spec needs both a linear-attention and a full-attention layer")
HID = X.HID
N_CORES = X.N_CORES
ELEM = X.ELEM
DENSE = X.KIND == "dense"
ONDV = X.ONDV
QKV_PC, Z_PC, OUT_PC = D.QKV_PC, D.Z_PC, D.OUT_PC
VW, OUT_K = D.VW, D.OUT_K
Q_PC, KV_PC, O_PC = DA.Q_PC, DA.KV_PC, DA.O_PC
QW, KVW, O_K = DA.QW, DA.KVW, DA.O_K
NH, KVH, HD = DA.NH, DA.KVH, DA.HD
NHL, RB = DA.NHL, DA.RB
ACORES = DA.ACORES
XN_ELEMS = D.XN_SIDE_ELEMS                                  # 1 on the 35B for both types
OG_ELEMS = D.OG_ELEMS                                       # 2 for both
PRE_BANDS = QKV_PC + Z_PC                                   # the unified body's pre-MoE band count
AX_BANDS = 2 * Q_PC + 2 * KV_PC                             # what a full-attention layer really needs
PAD_BANDS = PRE_BANDS - AX_BANDS                            # the dummy bands the ax sequence adds
KIND_LINEAR, KIND_FULL = 0, 1
KIND = int(os.environ.get("LAX_KIND", KIND_LINEAR))
# NLAYERS > 1 builds ONE control text that runs N whole layers back to back (the buffers are
# re-used, so layer k's output feeds layer k+1). This is the workaround for the measured
# per-hw_context limit of 8 whole-layer RUNS (the leak is per RUN, not per layer): with
# NLAYERS=5, 8 runs cover the 35B's 40 layers in ONE xrt::runlist submit.
NLAYERS = int(os.environ.get("LAX_NLAYERS", "1"))
# Per-layer kind pattern for a multi-layer text: 'l' = linear-attention, 'f' = full-attention.
# The 35B is a 3:1 hybrid, so NLAYERS=8 with pattern "lllf" gives [L,L,L,F,L,L,L,F] and 5 runs
# cover all 40 layers in the model's order. Empty -> all layers of the build's `kind`.
_PAT = os.environ.get("LAX_PATTERN", "")

def _layer_kinds(n, kind):
    if _PAT:
        m = {"l": KIND_LINEAR, "f": KIND_FULL}
        p = [m[ch] for ch in _PAT]
        return [p[i % len(p)] for i in range(n)]
    return [kind] * n

# dn_glue's head count, passed only when it differs from the header default (see lx.py)
GLUE_NHEAD_DEFAULT = 32
GLUE_FLAGS = {} if D.NHEAD == GLUE_NHEAD_DEFAULT else {"compile_flags": [f"-DDNGLUE_NHEAD={D.NHEAD}"]}

from recipes.attnknobs import probe_env  # noqa: E402
ATTN_FLAGS = [f"-DATTN_NH={NH}", f"-DATTN_KVH={KVH}", f"-DATTN_HD={HD}", f"-DATTN_ROT={DA.ROT}",
              "-DATTN_GATE=1", f"-DATTN_VEXP={DA.VEXP}", f"-DATTN_NHL={NHL}"]
if RB > 1:
    ATTN_FLAGS.append(f"-DATTN_RB={RB}")
for _k, _v in probe_env().items():
    if _k not in ("ATTN_RB", "ATTN_FAST"):
        ATTN_FLAGS.append(f"-D{_k}={_v}")


def rows3(t: int):
    """Tile t of each of the conv-state rows, in BYTES of the state BO (lx.py's helper)."""
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1, STATE_BYTES), t * D.TILE * 2, [1, 1, SPEC.conv_kernel - 1, D.TILE * 2],
                              [0, 0, D.NCH * 2, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"] + (["--generate-ctrl-pkt-overlay"] if os.environ.get("ONDV_CTRL_OVERLAY") == "1" else []))
def lax(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, state: InOut, cfg: In, *,
        kind: CompileTime[int] = KIND_LINEAR, srchash: CompileTime[int] = 0):
    return _lax_build(pool, xres, consts, kv, act, ptab, state, cfg, kind=kind, srchash=srchash,
                      ondv=ONDV, mrow=2, hrow=3)


def _lax_build(pool, xres, consts, kv, act, ptab, state, cfg, *, kind=KIND_LINEAR, srchash=0, ondv=False,
               mrow=2, hrow=3):
    t = X.types()
    tl = X.ln_types()
    u8_4k = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    u8_2k = np.ndarray[(2048,), np.dtype[np.uint8]]
    u8_ln = u8_4k
    u8_1k = np.ndarray[(DA.E_A,), np.dtype[np.uint8]]
    b256 = np.ndarray[(HD,), np.dtype[bfloat16]]
    b512 = np.ndarray[(KVW,), np.dtype[bfloat16]]
    f64 = np.ndarray[(DA.ROT,), np.dtype[np.float32]]
    f256 = np.ndarray[(HD,), np.dtype[np.float32]]
    f4096 = np.ndarray[(QW,), np.dtype[np.float32]]
    fq = np.ndarray[(2 * QW,), np.dtype[bfloat16]] if DA.VEXP else f4096
    fml = np.ndarray[(2 * DA.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    pb_ty = np.ndarray[(8 if RB > 1 else 4,), np.dtype[np.int32]]
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(C_BYTES,), np.dtype[np.uint8]]
    consts_a_ty = np.ndarray[(CA_BYTES,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(A_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(KV_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]
    nw_ty = np.ndarray[(SPEC.lin_value_dim,), np.dtype[bfloat16]]
    f32 = np.ndarray[(D.NHEAD,), np.dtype[np.float32]]
    fqk = np.ndarray[(2 * D.KEY_WIDTH,), np.dtype[np.float32]]
    fvt = np.ndarray[(D.TILE,), np.dtype[np.float32]]
    fxn = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(X.LN), str(X.LINL), str(X.RT), str(ATTN),
                            str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    if ondv:
        f_oc = ExternalFunction("ondv_ctrl_col", source_file=str(X.RT / "ondv_ctrl_col.cc"),
                                arg_types=[t["x"], t["x"], np.int32, tl["u8_ctrl"]],
                                include_dirs=inc + [str(X.RT)],
                                compile_flags=[f"-DONDV_EMIT_SHARED={int(X.ONDV_EMIT_SHARED)}",
                                               f"-DONDV_FIX_EXPERT={int(os.environ.get('ONDV_FIX_EXPERT', '0'))}",
                                               f"-DONDV_BD_UP={X.ONDV_BD_UP}",
                                               f"-DONDV_BD_GATE={X.ONDV_BD_GATE}",
                                               f"-DONDV_BD_DOWN={X.ONDV_BD_DOWN}"])

    DBG = ondv and os.environ.get("ONDV_DBG") == "1"
    dbg_ty = np.ndarray[(64,), np.dtype[np.uint32]]
    if DBG:
        f_echo = ExternalFunction("ondv_echo", source_file=str(X.RT / "ondv_echo.cc"),
                                  arg_types=[t["x"], t["x"], tl["u8_ctrl"], dbg_ty], include_dirs=inc)

    def af(sym, args):
        return ExternalFunction(sym, source_file=str(ATTN / f"{sym}.cc"), arg_types=args,
                                include_dirs=inc, compile_flags=ATTN_FLAGS)

    h0_arg = [np.int32] if ACORES > 1 else []
    f_meta = af("attn_meta", [u8_1k, u8_1k, b256, b256, f64, pb_ty])
    f_q = af("attn_q", [u8_1k, b256, f64, fq, np.int32])
    f_k = af("attn_k", [u8_1k, b256, f64, f256, b512, np.int32])
    f_v = af("attn_v", [u8_1k, b512, np.int32])
    f_init = af("attn_init", [foacc, fml])
    f_step = af("attn_step", [u8_1k, u8_1k, fq, foacc, fml, pb_ty] + h0_arg)
    f_stepn = af("attn_step_new", [b512, b512, fq, foacc, fml] + h0_arg)
    f_fin = af("attn_fin", [foacc, fml, u8_1k, u8_1k, b512, np.int32])
    f_ab = ExternalFunction("glue_ab", source_file=str(GLUE / "glue_ab.cc"),
                            arg_types=[u8_4k, fxn, f32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_small = ExternalFunction("glue_small_fn", source_file=str(GLUE / "glue_small.cc"),
                               arg_types=[u8_4k, f32, f32, f32, f32], include_dirs=inc, **GLUE_FLAGS)
    f_conv = ExternalFunction("glue_conv", source_file=str(GLUE / "glue_conv.cc"),
                              arg_types=[u8_2k, u8_2k, u8_2k, u8_2k, u8_2k, u8_4k, u8_4k, u8_2k, u8_2k, u8_2k,
                                         fqk, fvt, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_emit = ExternalFunction("glue_emit_fn", source_file=str(GLUE / "glue_emit.cc"),
                              arg_types=[fqk, fvt, f32, f32, u8_2k, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_copy = ExternalFunction("glue_copy_xn", source_file=str(GLUE / "glue_copy.cc"),
                              arg_types=[u8_4k, fxn], include_dirs=inc, **GLUE_FLAGS)
    post_fn = ExternalFunction("post_fn", source_file=str(POST / "post.cc"),
                               arg_types=[u8_4k, u8_4k, nw_ty, u8_2k], include_dirs=inc)
    post_copy = ExternalFunction("post_copy_nw", source_file=str(POST / "post_copy.cc"),
                                 arg_types=[u8_4k, nw_ty], include_dirs=inc)

    # ---- fifos
    # depth >= a routed band (8 elements) so a pinned routed descriptor, which transfers
    # a whole 81920-B band in ONE BD, fits the fifo the way the one-emitter probe's
    # single-element descriptor fits its depth-1 fifo (ONDV_W_DEPTH overrides)
    of_w = [ObjectFifo(t["elem"], name=f"w{c}", depth=int(os.environ.get("ONDV_W_DEPTH", 2)))
            for c in range(N_CORES)]
    of_y = [ObjectFifo(t["y"], name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(t["x"], name="x", depth=2)
    of_dbg = ObjectFifo(dbg_ty, name="dbg", depth=1) if DBG else None
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=3)
    of_side = ObjectFifo(u8_4k, name="side", depth=2)
    of_gact = ObjectFifo(u8_2k, name="gact", depth=5)
    of_gout = ObjectFifo(u8_2k, name="gout", depth=3)
    of_pin = ObjectFifo(u8_4k, name="pin", depth=2)
    of_pout = ObjectFifo(u8_2k, name="pout", depth=2)
    # the full-attention helper set (ax.py)
    of_ain = ObjectFifo(u8_1k, name="ain", depth=max(4, 2 * RB + 2))
    of_aout = ObjectFifo(b512, name="aout", depth=2)
    of_og = [ObjectFifo(b512, name=f"og{c}", depth=2) for c in range(1, ACORES)]
    emitter_tile = [Tile(c, int(os.environ.get("ONDV_EMITTER_ROW", 4)), tile_type=AIETileType.CoreTile)
                    for c in range(N_CORES)] if ondv else None
    shim_w = [Tile(c, 0, tile_type=AIETileType.ShimNOCTile) for c in range(N_CORES)]
    ctrlw = [Buffer(tl["u8_ctrl"], name=f"ctrlw{c}", tile=emitter_tile[c]) for c in range(N_CORES)] if ondv else None
    pktlk = [[Lock(emitter_tile[c], init=0, name=f"pktlk{c}_{i}") for i in range(X.NE + 1 + X.ONDV_EMIT_SHARED)] for c in range(N_CORES)] if ondv else None
    pktdone = [Lock(emitter_tile[c], init=0, name=f"pktdone{c}") for c in range(N_CORES)] if ondv else None

    def rep(fn):
        """Runtime-loop a per-layer core body NLAYERS times (NOT unrolled: the main core is
        near its 16 KB program limit, so a Python loop would overflow it)."""
        if NLAYERS == 1:
            return fn

        def g(*a):
            for _ in range_(NLAYERS):
                fn(*a)
        return g

    # ---- the unified main core: lx.py's body verbatim (role is inert on an all-q4_1 spec)
    def main_body(win, xin, yout, *args):
        B, K = X.unpack_args(args)
        tab = B["tab"]
        xe = xin.acquire(1)
        K["prep2048"](xe, tab)
        X.role_gemv_bands(win, yout, B, K, "linear", PRE_BANDS, HID)
        xin.release(1)
        X.dn_body(win, yout, B, K)
        oe = xin.acquire(2)
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
            for tile in range_(D.AB_ELEMS):
                ww = sin.acquire(1)
                fab(ww, xn, acc, tile)
                sin.release(1)
        sm = sin.acquire(1)
        fsmall(sm, acc_a, acc_b, decay, beta)
        sin.release(1)
        for base, ntiles in ((0, D.VALUE_TILE0), (D.VALUE_TILE0, D.NT - D.VALUE_TILE0)):
            for tt in range_(ntiles):
                ww = sin.acquire(CONVW_ELEMS)
                e = ain.acquire(2 + CONV_ROWS)
                o = oout.acquire(CONV_ROWS)
                fconv(e[0], e[1], e[2], e[3], e[4], ww[0], ww[1], o[0], o[1], o[2], qk, vt, tt, base)
                oout.release(CONV_ROWS)
                ain.release(2 + CONV_ROWS)
                sin.release(CONVW_ELEMS)
                if base == D.VALUE_TILE0:
                    for i in range_(D.HEADS_PER_TILE):
                        r = oout.acquire(1)
                        femit(qk, vt, decay, beta, r, tt, i)
                        oout.release(1)

    def post_body(ain, aout, nwb, f, fc):
        e = ain.acquire(1)
        fc(e, nwb)
        ain.release(1)
        for _ in range_(D.NG):
            e = ain.acquire(2)
            r = aout.acquire(1)
            f(e[0], e[1], nwb, r)
            aout.release(1)
            ain.release(2)

    N_OG = NHL // DA.HPO

    def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
              fm, fqk, fkk, fv, fi, fs, fsn, ff, c):
        h0 = c * NHL
        e = ain.acquire(2)
        fm(e[0], e[1], qn, kn, cs, pb)
        ain.release(2)
        for h in range_(DA.Q_AIN_ELEMS):
            e = ain.acquire(1)
            fqk(e, qn, cs, qs, h)
            ain.release(1)
        for h in range_(DA.K_AIN_ELEMS):
            e = ain.acquire(1)
            fkk(e, kn, cs, tmp, kout, h)
            ain.release(1)
        for h in range_(DA.K_AIN_ELEMS):
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

    def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                  fm, fqk, fkk, fv, fi, fs, fsn, ff):
        _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fqk, fkk, fv, fi, fs, fsn, ff, 0)

    def make_attn_body(c):
        def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                 fm, fqk, fkk, fv, fi, fs, fsn, ff):
            _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, fm, fqk, fkk, fv, fi, fs, fsn, ff, c)
        return body

    workers = [Worker(rep(X.ln_router_body),
                      fn_args=[of_lni.cons(), of_lno.prod(), Buffer(tl["xb"], name="rxs"),
                               Buffer(tl["racc"], name="racc"), L["ln_nr"], L["ln"], L["rcopy"], L["racc"],
                               L["rfin"]], tile=Tile(0, hrow), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(rep(main_body),
                              fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                       *X.worker_args(X.core_buffers(t, c), K)],
                              tile=Tile(c, mrow), stack_size=0x1800))
    workers.append(Worker(rep(post_body), fn_args=[of_pin.cons(), of_pout.prod(), Buffer(nw_ty, name="nwb"),
                                              post_fn, post_copy], tile=Tile(1, hrow), stack_size=0x1800))
    workers.append(Worker(rep(glue_body),
                          fn_args=[of_side.cons(), of_gact.cons(), of_gout.prod(),
                                   Buffer(f32, name="acc_a"), Buffer(f32, name="acc_b"),
                                   Buffer(f32, name="decay"), Buffer(f32, name="beta"),
                                   Buffer(fqk, name="qk"), Buffer(fvt, name="vt"), Buffer(fxn, name="xnb"),
                                   f_ab, f_small, f_conv, f_emit, f_copy],
                          tile=Tile(2, hrow), stack_size=0x1800))

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(b256, name=f"qn{s}"), Buffer(b256, name=f"kn{s}"), Buffer(f64, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(f256, name=f"tmp{s}"), Buffer(b512, name=f"kout{s}"),
                Buffer(b512, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]

    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin]
    # the attention cores live at row hrow AFTER the glue core (Tile(2)): (3, hrow) .. (2+ACORES, hrow)
    workers.append(Worker(rep(attn_body), fn_args=[of_ain.cons(), of_aout.prod()] + abufs(0) + afns,
                          tile=Tile(3, hrow), stack_size=0x1800))
    for c in range(1, ACORES):
        workers.append(Worker(rep(make_attn_body(c)), fn_args=[of_ain.cons(), of_og[c - 1].prod()] + abufs(c) + afns,
                              tile=Tile(3 + c, hrow), stack_size=0x1800))

    if ondv:
        N_X_SKIP = XN_ELEMS + OG_ELEMS + 1
        PKD_ACQ = os.environ.get("ONDV_PKTDONE_ACQ") == "1"

        def _emitter_body(c):
            def emitter_body(xin, ctrl, f_oc_col, *locks):
                if PKD_ACQ:
                    pkd, locks = locks[-1], locks[:-1]
                if DBG and c == 0:
                    f_echo_, dbg_out = locks[0], locks[1]
                    locks = locks[2:]
                for _ in range_(N_X_SKIP):
                    e = xin.acquire(1)
                    xin.release(1)
                # acquire(2), NOT two acquire(1)s: object-fifo acquire counts are cumulative, so a
                # second acquire(1) while holding one element returns that SAME element -- the
                # emitter then read the router output as its pool base (the "cfg[2..9] not
                # delivered" and page-fault symptoms)
                rc = xin.acquire(2)
                r, cfg = rc[0], rc[1]
                f_oc_col(r, cfg, c, ctrl)
                if DBG and c == 0:
                    de = dbg_out.acquire(1)
                    f_echo_(r, cfg, ctrl, de)
                    dbg_out.release(1)
                xin.release(2)
                locks[0].release(1)
                for e in range(X.NX):       # NX = NE+1: consume the shared expert's h too
                    h = xin.acquire(1)
                    xin.release(1)
                    if e < X.NE or X.ONDV_EMIT_SHARED:   # routed slots (and the shared one's down)
                        locks[e + 1].release(1)
                if PKD_ACQ:
                    # take back every chunk BD's pktdone release: otherwise the lock only counts up
                    # (one per chunk per layer) and AIE2 locks are 6-bit
                    pkd.acquire(X.NE + 1 + X.ONDV_EMIT_SHARED)
            return emitter_body

        for c in range(N_CORES):
            extra = [f_echo, of_dbg.prod()] if (DBG and c == 0) else []
            workers.append(Worker(rep(_emitter_body(c)), fn_args=[of_x.cons(), ctrlw[c], f_oc, *extra, *pktlk[c]] + ([pktdone[c]] if PKD_ACQ else []),
                                  tile=emitter_tile[c], stack_size=0x1800))

    bt = X.bt
    BB_HID, BB_OUT = X.role_band_bytes("linear", HID), X.role_band_bytes("linear_out", OUT_K)
    BB_AH, BB_AO = X.role_band_bytes("attn", HID), X.role_band_bytes("attn", O_K)
    YB = X.BAND_ROWS * 4
    AB_ELEMS = D.AB_ELEMS
    CONV_ROWS = SPEC.conv_kernel - 1
    CONVW_ELEMS = SPEC.conv_kernel * D.TILE * 2 // ELEM
    AB_TILES = [min(ELEM // 2, HID - h * (ELEM // 2)) // 64 for h in range(XN_ELEMS)]

    # ---- host sequences (one per instruction stream)
    def lx_sequence(a_pool, c_xres, a_consts, a_state, a_act, a_cfg, lni, lno, w_prods, x_prod, y_conss,
                    side_p, gact_p, gout_c, pin_p, pout_c, configure_routed=True):
        """The linear-attention stream (lx.py's, verbatim)."""
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(C_BYTES, C_LNW, ELEM), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(A_BYTES, A_XN, ELEM), wait=True, group=tg_ln)
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
            py.drain(y_conss[c], a_act, bt(A_BYTES, A_QKV + c * QKV_PC * YB, QKV_PC * YB))
            py.drain(y_conss[c], a_act, bt(A_BYTES, A_Z + c * Z_PC * YB, Z_PC * YB))
        tg_ln.finish()
        px = Pipeline(3)
        px.fill(x_prod, a_act, bt(A_BYTES, A_XN, ELEM))
        tg_s = TaskGroup()
        side_p.fill(a_act, tap=bt(A_BYTES, A_XN, ELEM), wait=True, group=tg_s)
        side_p.fill(a_consts, tap=bt(C_BYTES, C_SIDE, GLUE_SIDE_BYTES), wait=True, group=tg_s)
        py.finish()
        pipe = Pipeline(3)
        for tt in range(D.NT):
            pipe.drain(gout_c, a_state, rows3(tt))
            if tt >= D.VALUE_TILE0:
                pipe.drain(gout_c, a_act, bt(A_BYTES, A_VEC + (tt - D.VALUE_TILE0) * D.HEADS_PER_TILE * D.RECORD_BYTES,
                                             D.HEADS_PER_TILE * D.RECORD_BYTES))
            pipe.fill(gact_p, a_act, bt(A_BYTES, A_QKV + tt * D.TILE * 4, D.TILE * 4))
            pipe.fill(gact_p, a_state, rows3(tt))
        pipe.finish()
        tg_s.finish()
        X.dn_sequence(pw, py, a_state, a_act, w_prods, y_conss, A_BYTES, A_VEC, A_O, STATE_BYTES,
                      STATE_S_OFF, S_HEAD_BYTES)
        py.finish()
        pipe = Pipeline(3)
        pipe.fill(pin_p, a_consts, bt(C_BYTES, C_NW, ELEM))
        for g in range(D.NG):
            pipe.drain(pout_c, a_act, bt(A_BYTES, A_OG + g * D.G * 2, D.G * 2))
            pipe.fill(pin_p, a_act, bt(A_BYTES, A_O + g * D.G * 4, D.G * 4))
            pipe.fill(pin_p, a_act, bt(A_BYTES, A_Z + g * D.G * 4, D.G * 4))
        pipe.finish()
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_consts, bt(C_BYTES, C_WOUT + c * OUT_PC * BB_OUT, OUT_PC * BB_OUT))
            py.drain(y_conss[c], a_act, bt(A_BYTES, A_OUT + c * OUT_PC * YB, OUT_PC * YB))
        px.fill(x_prod, a_act, bt(A_BYTES, A_OG, OG_ELEMS * ELEM))
        py.finish()
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
        X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods,
                       x_prod, y_conss, A_BYTES, C_BYTES, A_XM, A_ROUT, A_RES, A_HP, C_SGW,
                       ondv=(a_cfg,), configure_routed=configure_routed)

    def w_regions(c):
        return [(POOL_Q + c * Q_PC * BB_AH, Q_PC * BB_AH), (POOL_GATE + c * Q_PC * BB_AH, Q_PC * BB_AH),
                (POOL_K + c * KV_PC * BB_AH, KV_PC * BB_AH), (POOL_V + c * KV_PC * BB_AH, KV_PC * BB_AH),
                (POOL_O + c * O_PC * BB_AO, O_PC * BB_AO)]

    def y_regions(c):
        qb, kb = Q_PC * YB, KV_PC * YB
        return [(AA_QG + c * qb, qb), (AA_QG + QW * 4 + c * qb, qb),
                (AA_KVN + c * kb, kb), (AA_KVN + KVW * 4 + c * kb, kb),
                (AA_OUT + c * O_PC * YB, O_PC * YB)]

    def ax_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, a_state, a_cfg,
                    lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c, og_cs, configure_routed=True):
        """The full-attention stream (ax.py's) against the UNIFIED main body: it fills
        `PRE_BANDS` pre-MoE bands (18 real + PAD_BANDS dummy) and, because the body always
        runs the DeltaNet step, feeds and sinks that step inside `state` (unused here)."""
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(CA_BYTES, CA_LNW, ELEM), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(AA_BYTES, AA_XN, ELEM), wait=True, group=tg_ln)
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            for off, n in w_regions(c)[:3]:
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, off, n))
            for off, n in y_regions(c)[:3]:
                py.drain(y_conss[c], a_act, bt(AA_BYTES, off, n))
        tg_ln.finish()
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(AA_BYTES, AA_XN, ELEM), wait=True, group=tg_x)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[3]))
            py.drain(y_conss[c], a_act, bt(AA_BYTES, *y_regions(c)[3]))
        # -- the unified body's extra pre-MoE bands, then its DeltaNet step (both into `state`) --
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Q, PAD_BANDS * BB_AH))
            py.drain(y_conss[c], a_state, bt(STATE_BYTES, 0, PAD_BANDS * YB))
        X.dn_sequence(pw, py, a_state, a_state, w_prods, y_conss, STATE_BYTES, STATE_S_OFF, STATE_S_OFF,
                      STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES)
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        pa_out.drain(aout_c, a_kv, bt(KV_BYTES, KV_ROW, KV_ROW))
        pa_out.drain(aout_c, a_act, bt(AA_BYTES, AA_OG, NHL * HD * 2))
        for c in range(1, ACORES):
            pa_out.drain(og_cs[c - 1], a_act, bt(AA_BYTES, AA_OG + c * NHL * HD * 2, NHL * HD * 2))
        pa_in.fill(ain_p, a_consts, bt(CA_BYTES, CA_META, DA.E_A))
        pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, PTAB_ROW, PTAB_ROW))
        py.finish(*y_conss)
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_QG, QW * 4))
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_KVN, KVW * 4))
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_KVN + KVW * 4, KVW * 4))
        pa_in.fill(ain_p, a_kv, bt(KV_BYTES, 0, KV_ROW))
        pa_in.fill(ain_p, a_act, bt(AA_BYTES, AA_QG + QW * 4, QW * 4))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *w_regions(c)[4]))
            py.drain(y_conss[c], a_act, bt(AA_BYTES, *y_regions(c)[4]))
        tg_ln2 = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(CA_BYTES, CA_POSTLN, ELEM), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(AA_BYTES, AA_RES, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(AA_BYTES, AA_XM, ELEM), wait=True, group=tg_ln2)
        pa_out.finish()
        x_prod.fill(a_act, tap=bt(AA_BYTES, AA_OG, QW * 2), wait=True, group=tg_x)
        py.finish()
        lni.fill(a_act, tap=bt(AA_BYTES, AA_OUT, HID * 4), wait=True, group=tg_ln2)
        tg_r = TaskGroup()
        lni.fill(a_consts, tap=bt(CA_BYTES, CA_RW, X.W_ELEMS * ELEM), wait=True, group=tg_r)
        lno.drain(a_act, tap=bt(AA_BYTES, AA_ROUT, ELEM), wait=True, group=tg_r)
        tg_ln2.finish()
        tg_r.finish()
        pw.finish()
        pa_in.finish()
        tg_x.finish()
        X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods,
                       x_prod, y_conss, AA_BYTES, CA_BYTES, AA_XM, AA_ROUT, AA_RES, AA_HP, CA_SGW,
                       ondv=(a_cfg,), configure_routed=configure_routed)

    def sequence(*a):
        (a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, a_state) = a[:7]
        a_cfg = a[7] if ondv else None
        rest = a[8:] if ondv else a[7:]
        dbg_c = rest[13] if DBG else None
        (lni, lno, w_prods, x_prod, y_conss, side_p, gact_p, gout_c, pin_p, pout_c,
         ain_p, aout_c, og_cs) = rest[:13]
        kinds = _layer_kinds(NLAYERS, kind)
        for k, kk in enumerate(kinds):
            if kk == KIND_FULL:
                ax_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, a_state, a_cfg,
                            lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c, og_cs,
                            configure_routed=(k == 0))
            else:
                lx_sequence(a_pool, c_xres, a_consts, a_state, a_act, a_cfg,
                            lni, lno, w_prods, x_prod, y_conss, side_p, gact_p, gout_c, pin_p, pout_c,
                            configure_routed=(k == 0))
        if DBG:
            tg_d = TaskGroup()                    # the emitter echo -> the (unused) kv argument
            dbg_c.drain(a_kv, tap=X.bt(KV_BYTES, 0, 256), wait=True, group=tg_d)
            tg_d.finish()

    rt_args = [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty, state_ty]
    if ondv:
        rt_args += [np.ndarray[(ELEM,), np.dtype[np.uint8]]]
    rt_args += [of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                [of_w[c].prod(tile=shim_w[c] if ondv else Tile(c, 0)) for c in range(N_CORES)],
                of_x.prod(tile=Tile(1, 0)),
                [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)), of_gout.cons(tile=Tile(2, 0)),
                of_pin.prod(tile=Tile(4, 0)), of_pout.cons(tile=Tile(1, 0)),
                of_ain.prod(tile=Tile(5, 0)), of_aout.cons(tile=Tile(6, 0)),
                [of_og[c].cons(tile=Tile(3 + c, 0)) for c in range(ACORES - 1)]]
    if DBG:
        rt_args += [of_dbg.cons(tile=Tile(7, 0))]
    rt = Runtime(sequence, rt_args)
    flows = []
    if ondv:
        for c in X.ONDV_EMITTER_COLS:
            for lk in pktlk[c]:
                rt.add_lock(lk)
            rt.add_lock(pktdone[c])
            bds = [Bd(buffer=ctrlw[c], offset=0, length=120,
                      acquires=[Acquire(pktlk[c][0])], releases=[Release(pktdone[c])], next=1)]
            for k in range(1, X.NE):
                bds.append(Bd(buffer=ctrlw[c], offset=180 * k - 60, length=180,
                              acquires=[Acquire(pktlk[c][k])], releases=[Release(pktdone[c])], next=k + 1))
            if X.ONDV_EMIT_SHARED:
                # down7 + the shared up|gate, then the shared down on its own after the shared h
                bds.append(Bd(buffer=ctrlw[c], offset=1380, length=180,
                              acquires=[Acquire(pktlk[c][X.NE])], releases=[Release(pktdone[c])], next=X.NE + 1))
                bds.append(Bd(buffer=ctrlw[c], offset=1560, length=60,
                              acquires=[Acquire(pktlk[c][X.NE + 1])], releases=[Release(pktdone[c])], next=0))
            else:
                bds.append(Bd(buffer=ctrlw[c], offset=1380, length=60,
                              acquires=[Acquire(pktlk[c][X.NE])], releases=[Release(pktdone[c])], next=0))
            rt.add_tile_dma(TileDma(tile=emitter_tile[c],
                                    channels=[DmaChannel(direction=DMAChannelDir.MM2S, channel=1, bds=bds)]))
            flows.append(PacketFlow(pkt_id=int(os.environ.get("ONDV_PKT_ID", "15")), src=emitter_tile[c], src_port=WireBundle.DMA, src_channel=1,
                                    dst=shim_w[c], dst_port=WireBundle.TileControl, dst_channel=0,
                                    keep_pkt_header=os.environ.get("ONDV_KEEP_HDR", "1") == "1")) if (os.environ.get("ONDV_NO_EMITTERS") != "1" or os.environ.get("ONDV_FORCE_FLOW") == "1") else None
    for f in flows:
        rt.add_flow(f)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = lax
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes()] + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.cc"))
                + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(),
                   (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"kind": KIND, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
