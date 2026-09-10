r"""dx_attn: Dispatch B of the three-dispatch T-token prefill split (#32, task
0167 stage 12) -- the SINGLE-TOKEN attention phase of dx.py's fused layer,
built as its OWN xclbin/Program, with NOTHING else (no LN core, no GEMV
cores, no pool/xres). Every kernel and every Python wiring line in this file
is copied VERBATIM from dx.py's t == 1 `sequence()` "3. attention" section
and its `_attn`/`attn_body`/`make_attn_body` definitions -- same C++ sources
(attn.h, attn_meta.cc, ...), same ATTN_FLAGS, same abufs()/afns, same
Pipeline(3) usage, same "placeholder position 1" offsets (`bt(L.KV_BYTES, 0,
L.KV_ROW)` for the window fill, `bt(L.KV_BYTES, L.KV_ROW, L.KV_ROW)` for the
row drain, `bt(L.PTAB_BYTES, L.PTAB_ROW, L.PTAB_ROW)` for the position
record) -- so a compiled dx_attn.xclbin is patchable by the EXISTING,
unmodified `stream_patch.hpp` / `run_kernel.cpp` `attnpos` directive exactly
like production decode's own full-layer dispatch already is.

Why this exists (see tasks/0167 Stage 11 in NpuEmbeddings for the full
account): a T-token BATCHED attention phase inside dx.py's own fused
`sequence_multi` hangs on real hardware at T=2 for a cause that survived an
exhaustive audit (block kernel, cached-row compute, the intra-dispatch KV
read-after-write, the six ungated per-token kernels, the acquire/release lock
protocol -- all exonerated). The certain, zero-new-risk fallback Stage 11
priced is T SEPARATE single-token attention dispatches reusing the
ALREADY-VALIDATED T=1 machinery verbatim, patched to each token's own
position via the SAME attnpos mechanism production decode already uses
thousands of times per process. This file IS that dispatch.

Args: consts (the layer's [lnw | postln | qn kn] blob -- only CD_META is
read), kv (the layer's KV cache row store, InOut: window read + new-row
write), act (InOut: reads AD_Q/AD_KVN -- q/k/v written by a prior Dispatch A
call into the SAME per-token slice of a T-wide act buffer -- and writes
AD_OG), ptab (the RoPE position table; the record row read is the
placeholder position 1, patched by attnpos before each of the T calls).
No `pool`, no `xres`: this dispatch does no GEMV and touches no residual.

Build (Windows, iron_env.ps1 dot-sourced): OPEN_KERNELS_SPEC=<spec>
python build_design.py designs/dense/dx_attn.py designs/dense/build_dx_attn
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
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes import dense as QR  # noqa: E402
from aie.helpers.taplib import TensorAccessPattern  # noqa: E402

SPEC = current_spec()
R = QR.recipe(SPEC)
L, G = R.layout, R.geo
E_A = L.E_A
QW, KVW = G.QW, G.KVW
HID = G.HID

# ---- verbatim from dx.py: ATTN_FLAGS / ACORES / NHL / RB (same derivation,
# same module-level constants dx.py's own attention kernels are compiled
# against -- a mismatched flag set would be a DIFFERENT kernel, not a reused
# one, so this is copied character-for-character rather than re-derived).
ATTN_FLAGS = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
              f"-DATTN_QKNORM={1 if G.QKNORM else 0}", f"-DATTN_QKNORM_POST={1 if G.QKNORM_POST else 0}",
              f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}", f"-DATTN_NHL={G.NHL}"]
if G.RB > 1:
    ATTN_FLAGS.append(f"-DATTN_RB={G.RB}")
for _k, _v in QR.probe_env().items():               # ATTN_NULL / ATTN_ABL, same as dx.py
    if _k != "ATTN_RB":
        ATTN_FLAGS.append(f"-D{_k}={_v}")
ACORES, NHL, RB = G.ACORES, G.NHL, G.RB


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dx_attn(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, srchash: CompileTime[int] = 0):
    # 0167 stage 12 / #32: `pool`/`xres` are UNUSED dummy args, never touched
    # by sequence_attn's body below. They exist ONLY to make `kv` land at
    # buffer-argument index 3 and `ptab` at index 5 -- the EXACT positions
    # stream_patch.hpp's attn_table()/attn_apply() hardcode (its own comment:
    # "the KV window fill is the patch on arg 3 ... the position record is
    # arg 5"), because that convention comes from dx.py's OWN Runtime
    # argument order (pool, xres, consts, kv, act, ptab) and is shared,
    # unparametrised, across every attnpos-patched design in this tree
    # (make_decode.py, layer_x/ax.py, decode_chain/make_27b.py all rely on
    # it). Found on hardware, not by inspection: a first cut with a lean
    # 4-arg signature (consts, kv, act, ptab) compiled and ran, but
    # `run_kernel.exe`'s `attnpos` directive threw "unexpected kv transfer at
    # offset 1024" -- attn_table() was reading the ptab fill (MY arg index 3)
    # as if it were a kv transfer, because in a 4-arg signature ptab lands at
    # index 3, not 5. The two dummy args are the fix; the caller's `run dxB`
    # line passes the SAME `pool`/`xres` buffers Dispatch A/C already
    # declared (harmless -- this dispatch configures no DMA against them at
    # all, confirmed by their absence from `sequence_attn` below).
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    pb_ty = np.ndarray[(8 if RB > 1 else 4,), np.dtype[np.int32]]
    bhd = np.ndarray[(G.HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(G.HD,), np.dtype[np.float32]]
    fq = (np.ndarray[(2 * QW,), np.dtype[bfloat16]]
          if G.VEXP else np.ndarray[(QW,), np.dtype[np.float32]])
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * G.HD,), np.dtype[np.float32]]
    i32 = np.int32
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]   # dummy -- see dx_attn()'s own comment
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]          # dummy -- see dx_attn()'s own comment
    consts_ty = np.ndarray[(L.CD_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(L.KV_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]   # ONE token's slice -- see docstring
    ptab_ty = np.ndarray[(L.PTAB_BYTES,), np.dtype[np.uint8]]

    inc = include_dirs() + [str(ATTN)]

    def ef(sym, src, args, flags=["-Os"]):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    # ---- verbatim from dx.py: the attention kernel declarations (same
    # symbols, same source files, same ATTN_FLAGS -- these ARE dx.py's own
    # f_meta..f_fin, not a reimplementation).
    f_meta = ef("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a, bhd, bhd, fcs, pb_ty], ATTN_FLAGS)
    f_q = ef("attn_q", ATTN / "attn_q.cc", [u8_a, bhd, fcs, fq, i32], ATTN_FLAGS)
    f_k = ef("attn_k", ATTN / "attn_k.cc", [u8_a, bhd, fcs, fhd, brow, i32], ATTN_FLAGS)
    f_v = ef("attn_v", ATTN / "attn_v.cc", [u8_a, brow, i32], ATTN_FLAGS)
    f_init = ef("attn_init", ATTN / "attn_init.cc", [foacc, fml], ATTN_FLAGS)
    h0_arg = [i32] if ACORES > 1 else []
    f_step = ef("attn_step", ATTN / "attn_step.cc", [u8_a, u8_a, fq, foacc, fml, pb_ty] + h0_arg, ATTN_FLAGS)
    f_stepn = ef("attn_step_new", ATTN / "attn_step_new.cc", [brow, brow, fq, foacc, fml] + h0_arg, ATTN_FLAGS)
    f_stepb = (ef("attn_stepb", ATTN / "attn_stepb.cc", [u8_a] * (2 * RB) + [fq, foacc, fml, pb_ty] + h0_arg,
                  ATTN_FLAGS) if RB > 1 else None)
    f_fin = ef("attn_fin_ng", ATTN / "attn_fin_ng.cc", [foacc, fml, brow, i32], ATTN_FLAGS)

    # ---- fifos, verbatim from dx.py
    of_ain = ObjectFifo(u8_a, name="ain", depth=max(4, 2 * RB + 2))
    of_aout = ObjectFifo(brow, name="aout", depth=2)
    of_og = [ObjectFifo(brow, name=f"og{c}", depth=2) for c in range(1, ACORES)]

    N_OG = NHL // G.HPO

    # ---- verbatim from dx.py: _attn / attn_body / make_attn_body
    def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
              f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, h0):
        e = ain.acquire(2)                                      # [qn | kn], the position record
        f_meta(e[0], e[1], qn, kn, cs, pb)
        ain.release(2)
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
        if aout is not None:                                    # core 0 owns the cache row
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
            for _ in range_(pb[4]):                             # whole blocks of RB rows
                e = ain.acquire(2 * RB)
                args = [e[i] for i in range(2 * RB)] + [qs, oacc, ml, pb] + ([h0] if ACORES > 1 else [])
                f_stepb(*args)
                ain.release(2 * RB)
            for _ in range_(pb[5]):                             # what did not fill a block
                e = ain.acquire(2)
                f_step(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else f_step(e[0], e[1], qs, oacc, ml, pb)
                ain.release(2)
        else:
            for _ in range_(pb[1]):                             # nf cached rows (K_t, V_t)
                e = ain.acquire(2)
                f_step(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else f_step(e[0], e[1], qs, oacc, ml, pb)
                ain.release(2)
        f_stepn(kout, vout, qs, oacc, ml, h0) if ACORES > 1 else f_stepn(kout, vout, qs, oacc, ml)
        for hp in range_(N_OG):
            o = ogout.acquire(1)
            f_fin(oacc, ml, o, hp)
            ogout.release(1)

    if RB > 1:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                  f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, 0)

        def make_attn_body(c):
            h0 = c * NHL
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                      f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, h0)
            return body
    else:
        def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin):
            _attn(ain, aout, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                  f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, None, 0)

        def make_attn_body(c):
            h0 = c * NHL
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin):
                _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                      f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, None, h0)
            return body

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s}"), Buffer(bhd, name=f"kn{s}"), Buffer(fcs, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(fhd, name=f"tmp{s}"), Buffer(brow, name=f"kout{s}"),
                Buffer(brow, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]

    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin] + ([f_stepb] if RB > 1 else [])

    workers = [Worker(attn_body, fn_args=[of_ain.cons(), of_aout.prod()] + abufs(0) + afns,
                      tile=Tile(2, 3), stack_size=0x1800)]
    for c in range(1, ACORES):
        workers.append(Worker(make_attn_body(c), fn_args=[of_ain.cons(), of_og[c - 1].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=0x1800))

    def sequence_attn(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, ain_p, aout_c, og_cs):
        # a_pool / c_xres: unused dummies, kept only for buffer-argument
        # position (see dx_attn()'s own comment).
        # 0167 stage 12 / #32: verbatim (minus the o-proj/GEMV lines, which
        # this dispatch does not do) copy of dx.py's t == 1 `sequence()`
        # section "3. attention" -- same Pipeline(3) objects, same offsets,
        # so the compiled instruction stream is patchable by the EXISTING
        # attnpos mechanism exactly like production decode's full-layer
        # dispatch already is (attn_table() wants exactly one patch of each
        # of its four kinds; this dispatch's shape supplies exactly that).
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        pa_out.drain(aout_c, a_kv, bt(L.KV_BYTES, L.KV_ROW, L.KV_ROW))          # [k' | v'] -> row pos (attnpos)
        pa_out.drain(aout_c, a_act, bt(L.AD_BYTES, L.AD_OG, NHL * G.HD * 2))
        for c in range(1, ACORES):                                          # heads NHL*c ..
            pa_out.drain(og_cs[c - 1], a_act, bt(L.AD_BYTES, L.AD_OG + c * NHL * G.HD * 2, NHL * G.HD * 2))
        pa_in.fill(ain_p, a_consts, bt(L.CD_BYTES, L.CD_META, E_A))            # [qn | kn]
        pa_in.fill(ain_p, a_ptab, bt(L.PTAB_BYTES, L.PTAB_ROW, L.PTAB_ROW))    # the position record (attnpos)
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_Q, QW * 4))
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN, KVW * 4))
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN + KVW * 4, KVW * 4))
        pa_in.fill(ain_p, a_kv, bt(L.KV_BYTES, 0, L.KV_ROW))                   # the window: rows [0, nf) (attnpos)
        pa_out.finish()                                           # og (and the new cache row) are in DDR
        pa_in.finish()

    rt = Runtime(sequence_attn,
                 [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty,
                  of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0)),
                  [of_og[c].cons(tile=Tile(3 + c, 0)) for c in range(ACORES - 1)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = dx_attn
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("dx_attn.py"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + sorted(f.read_bytes() for f in (HERE.parent.parent / "recipes").glob("*.py"))
                + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
