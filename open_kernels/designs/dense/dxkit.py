"""The dense layer's attention cores, norm helper and host sequence, for one layer kind's geometry.

gx.py (Gemma 4 E-series) builds every kind of a spec into one image, so each piece here takes the kind's
DenseGeometry / DenseLayout instead of reading module globals. These are dx.py's pieces as of 2026-10-20,
when the E-series kernels were validated; dx.py itself has moved on since and does not use them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import ObjectFifo, TaskGroup, Worker, Buffer
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ironutil import Pipeline  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM  # noqa: E402
from recipes.dense import qkv_segments  # noqa: E402

HERE = Path(__file__).parent
ATTN = HERE.parent / "attn"
LN = HERE.parent / "ln"
LINL = HERE.parent / "lin_layer"
OS = ["-Os"]


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def acq(fifo, n):
    e = fifo.acquire(n)
    return [e] if n == 1 else e            # acquire(1) yields the element itself


def attn_flags(G, probe_env) -> list[str]:
    f = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
         f"-DATTN_QKNORM={1 if G.QKNORM else 0}", f"-DATTN_QKNORM_POST={1 if G.QKNORM_POST else 0}",
         f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}", f"-DATTN_NHL={G.NHL}"]
    if G.RB > 1:                                           # attn.h defaults it to 1; adding the flag
        f.append(f"-DATTN_RB={G.RB}")                      # would change every other family's build line
    if G.QKVB:                                             # same: only a family that has a bias sees the flag
        f.append("-DATTN_QKV_BIAS=1")
    if G.PTAB_ELEMS > 1:
        f.append("-DATTN_PTAB_SPLIT=1")
    # Gemma 4 E-series attention (recipes/gemma4e.py); the 12B builds through dx.py and dense.attn_flags
    if G.PTAB_CS_OFF != 512:
        f.append(f"-DATTN_PTAB_CS_OFF={G.PTAB_CS_OFF}")
    if G.SCALE:
        f.append(f"-DATTN_SCALE={G.SCALE!r}f")
    if G.VNORM:
        # soft float put the 512-wide core over program memory; the bit-identical integer routines fit
        f += ["-DATTN_VNORM=1", "-DATTN_INTFP=1", "-DATTN_NORM_SHARED=1"]
    if G.ROPE_PAIR:
        f.append(f"-DATTN_ROPE_PAIR={G.ROPE_PAIR}")
    if G.KV_SHARED and G.KVH == 1:                         # this position's row is one fifo element (attn_step_new.cc)
        f.append("-DATTN_KV_SHARED=1")
    if G.QLOCAL:
        raise ValueError("dxkit: ATTN_QLOCAL is the 12B's (dx.py); no E-series kind needs it")
    for _k, _v in probe_env.items():                       # ATTN_NULL / ATTN_ABL: see attn.h.
        if _k not in ("ATTN_RB", "ATTN_FAST", "OPEN_LAYER_ONE_CTX"):   # RB: via G.RB; FAST picks G; the layout is no attn flag.
            f.append(f"-D{_k}={_v}")                       # In the build key -- recipes/cache.py.
    return f


def ln_flags(G, norm_groups) -> list[str]:
    f = [f"-DLN_N={G.HID}", f"-DLN_EPS={G.EPS:g}f"]
    if norm_groups != 1:                       # ln.h defaults it to 1; adding the flag would change
        f.append(f"-DLN_GROUPS={norm_groups}")   # every other family's build line (K2: 2)
    return f


# ---- the norm helper (Tile(0, 3))
def ln_kernels(L, G, inc, flags):
    u8_ln = np.ndarray[(L.ELN,), np.dtype[np.uint8]]
    i32 = np.int32

    def ef(sym, src, args):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)
    f_nr = ef("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4)
    f_lny = ef("ln_y", LN / "ln_y.cc", [u8_ln] * 5 + [i32])
    f_lnx = ef("ln_xn", LN / "ln_xn.cc", [u8_ln] * 6)
    f_nr32 = ef("ln_nr32", LN / "ln_nr32.cc", [u8_ln] * 4 + [i32]) if G.SANDWICH else None
    return f_nr, f_lny, f_lnx, f_nr32


def ln_fifos(L):
    u8_ln = np.ndarray[(L.ELN,), np.dtype[np.uint8]]
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)      # one output element at a time (8 KB elements at 4096 wide)
    return of_lni, of_lno


def ple_kernels(L, inc, flags, d):
    """gx.py's per-layer input stages on the norm helper (designs/ln/ple_ln.cc); with no per-layer input
    (d = 0, the 12B) only the layer's output scale."""
    u8_ln = np.ndarray[(L.ELN,), np.dtype[np.uint8]]
    if not d:
        return (ExternalFunction("ple_scale", source_file=str(LN / "ple_ln.cc"), arg_types=[u8_ln] * 4 + [np.int32],
                                 include_dirs=inc, compile_flags=flags + ["-DPLE_SCALE=1"]),)
    f = flags + [f"-DPLE_D={d}"]
    f_act = ExternalFunction("ple_act", source_file=str(LN / "ple_ln.cc"), arg_types=[u8_ln, u8_ln], include_dirs=inc,
                             compile_flags=f)
    f_out = ExternalFunction("ple_out", source_file=str(LN / "ple_ln.cc"), arg_types=[u8_ln] * 6 + [np.int32],
                             include_dirs=inc, compile_flags=f + ["-DPLE_OUT=1"])
    return f_act, f_out


def make_ln_body(G, stop, ple=0):
    """`ple`: how many per-layer input kernels follow f_nr32 -- 0 none, 1 the output scale alone, 2 the full stage."""
    def ln_body(ain, aout, f_nr, f_lny, f_lnx, *rest):
        # 1. the layer-entry norm: [x0 x1 lnw] -> [xn]
        e = ain.acquire(3)
        o = aout.acquire(1)
        f_nr(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)

        def post_norm(f_nr32):
            # a sandwich norm on a block output: [o0 o1 w] -> [t0] [t1] (fp32 halves)
            e = ain.acquire(3)
            for i in range(2):
                o = aout.acquire(1)
                f_nr32(e[0], e[1], e[2], o, i)
                aout.release(1)
            ain.release(3)

        def add_norm(with_xn):
            # [x0 x1 w a0 a1] -> [y0] [y1] [xn]
            e = ain.acquire(5)
            for i in range(2):
                o = aout.acquire(1)
                f_lny(e[0], e[1], e[3], e[4], o, i)
                aout.release(1)
            if with_xn:
                o = aout.acquire(1)
                f_lnx(e[0], e[1], e[3], e[4], e[2], o)
                aout.release(1)
            ain.release(5)

        if G.SANDWICH:
            f_nr32 = rest[0]
            if stop >= 3:                      # group 2 runs from the o projection on
                post_norm(f_nr32)              # 2a. t = post_attn_norm(out)
                add_norm(True)                 # 2b. res = x + t; xm = pre_ffn_norm(res)
            if stop >= 99:                     # group 3 only in the full layer
                post_norm(f_nr32)              # 3a. t2 = post_ffn_norm(out2)
                add_norm(False)                # 3b. xres = res + t2
            if ple == 1:                       # the layer's output scale alone (the 12B in gx.py)
                e = ain.acquire(3)             # 4. [y0 y1 s] -> xres = y * s
                for i in range(2):
                    o = aout.acquire(1)
                    rest[1](e[0], e[1], e[2], o, i)
                    aout.release(1)
                ain.release(3)
            elif ple:                          # Gemma 4's per-layer input (gx.py)
                f_pact, f_pout = rest[1], rest[2]
                e = ain.acquire(1)             # 4a. [g | in] -> z = gelu_tanh(g) * in
                o = aout.acquire(1)
                f_pact(e, o)
                aout.release(1)
                ain.release(1)
                post_norm(f_nr32)              # 4b. r = post_norm(proj(z))
                e = ain.acquire(5)             # 4c. [y0 y1 s r0 r1] -> xres = (y + r) * s
                for i in range(2):
                    o = aout.acquire(1)
                    f_pout(e[0], e[1], e[2], e[3], e[4], o, i)
                    aout.release(1)
                ain.release(5)
        else:
            if stop >= 3:                      # group 2 runs from the o projection on
                add_norm(True)                 # 2. res = x + out; xm = post_attn_norm(res)
            if stop >= 99:                     # group 3 only in the full layer
                add_norm(True)                 # 3. xres = res + out2 (the xn is junk)
    return ln_body


def ln_worker(G, stop, of_lni, of_lno, lnk, tile=Tile(0, 3), ple=None):
    """`ple`: ple_kernels()'s kernels, for gx.py's per-layer input stages (sandwich norms only)."""
    f_nr, f_lny, f_lnx, f_nr32 = lnk
    assert ple is None or G.SANDWICH
    return Worker(make_ln_body(G, stop, len(ple) if ple else 0),
                  fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_lny, f_lnx] + ([f_nr32] if G.SANDWICH else []) + list(ple or []),
                  tile=tile, stack_size=0x1800)


# ---- the attention cores
class Attn:
    """One layer kind's attention cores: kernels, fifos and workers. `prefix` renames the kernel symbols
    and `sfx` the fifos and buffers, so several kinds can share one image (gx.py); dx.py passes neither."""

    def __init__(self, G, L, flags, inc, prefix=None, sfx=""):
        self.G, self.L, self.flags, self.inc, self.prefix, self.sfx = G, L, flags, inc, prefix, sfx
        self.SHARED, self.QKVB, self.RB = G.KV_SHARED, G.QKVB, G.RB
        self.NPTAB, self.CSE = G.PTAB_ELEMS, G.PTAB_CS_ELEM
        self.ACORES, self.NHL = G.ACORES, G.NHL
        self.ROW = 1 if G.KVH == 1 else 2                  # ain elements per cached row (attn.h's ATTN_KVH 1)
        self.OGH = min(G.NHL, G.HPO)                       # heads in one og element (attn.h's kOGH)
        self.N_OG = G.NHL // self.OGH                      # og elements a core emits
        KVW, HD, QW, RB = G.KVW, G.HD, G.QW, G.RB
        E_A = L.E_A
        self.u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
        self.u8_ab = np.ndarray[(E_A // 2,), np.dtype[np.uint8]]    # the same heads of a bf16 bias
        self.pb_ty = np.ndarray[(8 if RB > 1 else 4,), np.dtype[np.int32]]   # the attention parameter block:
                                                                             # [pos, nf, seen, -] + [blocks, remainder] when blocked
        self.bhd = np.ndarray[(HD,), np.dtype[bfloat16]]
        self.brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
        self.og_ty = np.ndarray[(self.OGH * HD,), np.dtype[bfloat16]]   # attn_fin writes kOGH heads at a time
        self.fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
        self.fhd = np.ndarray[(HD,), np.dtype[np.float32]]
        self.fq = (np.ndarray[(2 * QW,), np.dtype[bfloat16]]   # ATTN_VEXP: q pre-split, [hi | lo]
                   if G.VEXP else np.ndarray[(QW,), np.dtype[np.float32]])
        self.fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]   # [m | l], stride MLS (padded when ATTN_VEXP)
        self.foacc = np.ndarray[(G.NHL * HD,), np.dtype[np.float32]]   # this core's heads only

    def kernels(self):
        G, u8_a, u8_ab, pb_ty, bhd, brow, og_ty = self.G, self.u8_a, self.u8_ab, self.pb_ty, self.bhd, self.brow, self.og_ty
        fcs, fhd, fq, fml, foacc, i32 = self.fcs, self.fhd, self.fq, self.fml, self.foacc, np.int32
        SHARED, QKVB, RB, ROW, ACORES = self.SHARED, self.QKVB, self.RB, self.ROW, self.ACORES
        kw = {"symbol_prefix": self.prefix} if self.prefix else {}

        def ef(sym, src, args):
            return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=self.inc,
                                    compile_flags=self.flags, **kw)
        pz = [u8_a] if self.NPTAB > 1 else []                   # the record's second element
        self.f_meta = ef("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a] + pz + [bhd, bhd, fcs, pb_ty])
        bz = [u8_ab] if QKVB else []                            # the bias element, when the family has one
        self.f_q = ef("attn_q", ATTN / "attn_q.cc", [u8_a] + bz + [bhd, fcs, fq, i32])
        self.f_k = None if SHARED else ef("attn_k", ATTN / "attn_k.cc", [u8_a] + bz + [bhd, fcs, fhd, brow, i32])
        self.f_v = None if SHARED else ef("attn_v", ATTN / "attn_v.cc", [u8_a] + bz + [brow, i32])
        self.f_init = ef("attn_init", ATTN / "attn_init.cc", [foacc, fml])
        h0_arg = [i32] if ACORES > 1 else []                       # only a split needs the head offset
        self.f_step = ef("attn_step", ATTN / "attn_step.cc", [u8_a] * ROW + [fq, foacc, fml, pb_ty] + h0_arg)
        # a KV-shared layer's own row is already in the cache: it arrives as fifo elements, not core scratch
        nrow = [u8_a] * ROW if SHARED else [brow, brow]
        self.f_stepn = ef("attn_step_new", ATTN / "attn_step_new.cc", nrow + [fq, foacc, fml] + h0_arg)
        self.f_stepb = (ef("attn_stepb", ATTN / "attn_stepb.cc", [u8_a] * (ROW * RB) + [fq, foacc, fml, pb_ty] + h0_arg)
                        if RB > 1 else None)
        self.f_fin = ef("attn_fin_ng", ATTN / "attn_fin_ng.cc", [foacc, fml, og_ty, i32])

    def fifos(self, join=False, join_tile=None):
        """`join`: the cores' og elements are gathered in a memory tile and drained as one (gx.py, whose
        four kinds would otherwise need 16 og drains)."""
        G, s = self.G, self.sfx
        self.of_ain = ObjectFifo(self.u8_a, name=f"ain{s}", depth=max(4, 2 * self.RB + 2, 1 + self.NPTAB + 1))   # a block is acquired at once
        # Attention over ACORES cores: heads are independent, so each core owns NHL of
        # them and drains its own og element (or, joined, one stream for all of them).
        # of_aout carries the two KV cache rows ONLY. It used to carry core 0's og as well,
        # which is what forced every og element to be KVW wide and so fixed the heads per
        # element at HPO -- and with it ACORES at NH/HPO. An og element is now the core's own
        # heads, and core 0 has a fifo like every other core.
        self.of_aout = None if self.SHARED else ObjectFifo(self.brow, name=f"aout{s}", depth=2)
        if join:
            assert self.N_OG == 1, "dxkit: a joined og drain takes one element per core"
            jt = np.ndarray[(self.ACORES * self.OGH * G.HD,), np.dtype[bfloat16]]
            self.of_ogj = ObjectFifo(jt, name=f"og{s}", depth=2)
            self.of_og = self.of_ogj.prod().join([c * self.OGH * G.HD for c in range(self.ACORES)], tile=join_tile,
                                                  obj_types=[self.og_ty] * self.ACORES, depths=[2] * self.ACORES,
                                                  names=[f"og{s}_{c}" for c in range(self.ACORES)])
        else:
            self.of_ogj = None
            self.of_og = [ObjectFifo(self.og_ty, name=f"og{c}{s}", depth=2) for c in range(self.ACORES)]
        # The bias stream. Its own fifo rather than more elements on `ain`: the core needs bias
        # element i beside projection element i, and one fifo would mean either interleaving the
        # fills (one DMA descriptor per element) or holding the whole bias in the core's L1.
        self.of_abias = ObjectFifo(self.u8_ab, name=f"abias{s}", depth=4) if self.QKVB else None

    def _body_fns(self):
        G, SHARED, RB, ACORES, NPTAB, CSE, ROW, N_OG = (self.G, self.SHARED, self.RB, self.ACORES, self.NPTAB,
                                                         self.CSE, self.ROW, self.N_OG)
        KVW = G.KVW

        def _attn_kv(ain, kn, cs, tmp, kout, vout, f_k, f_v, bias_in):
            for h in range_(G.K_AIN_ELEMS):
                e = ain.acquire(1)
                if bias_in is None:
                    f_k(e, kn, cs, tmp, kout, h)
                else:
                    b = bias_in.acquire(1)
                    f_k(e, b, kn, cs, tmp, kout, h)
                    bias_in.release(1)
                ain.release(1)
            for h in range_(G.K_AIN_ELEMS):
                e = ain.acquire(1)
                if bias_in is None:
                    f_v(e, vout, h)
                else:
                    b = bias_in.acquire(1)
                    f_v(e, b, vout, h)
                    bias_in.release(1)
                ain.release(1)

        def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                  f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, h0, bias_in=None):
            # [qn | kn] then the position record, which is NPTAB elements wide: every family
            # until Qwen2.5-3B had exactly one, and acquiring fewer elements than the fill
            # delivers would leave the rest to be read as q.
            e = ain.acquire(1 + NPTAB)
            if NPTAB > 1:
                f_meta(e[0], e[1], e[1 + CSE], qn, kn, cs, pb)
            else:
                f_meta(e[0], e[1], qn, kn, cs, pb)
            ain.release(1 + NPTAB)
            for h in range_(G.Q_AIN_ELEMS):
                e = ain.acquire(1)
                if bias_in is None:
                    f_q(e, qn, cs, qs, h)
                else:
                    b = bias_in.acquire(1)
                    f_q(e, b, qn, cs, qs, h)
                    bias_in.release(1)
                ain.release(1)
            if not SHARED:
                _attn_kv(ain, kn, cs, tmp, kout, vout, f_k, f_v, bias_in)
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
                    e = acq(ain, ROW * RB)
                    args = [e[i] for i in range(ROW * RB)] + [qs, oacc, ml, pb] + ([h0] if ACORES > 1 else [])
                    f_stepb(*args)
                    ain.release(ROW * RB)
                for _ in range_(pb[5]):                             # what did not fill a block
                    e = acq(ain, ROW)
                    row = [e[i] for i in range(ROW)]
                    f_step(*row, qs, oacc, ml, pb, h0) if ACORES > 1 else f_step(*row, qs, oacc, ml, pb)
                    ain.release(ROW)
            else:
                for _ in range_(pb[1]):                             # nf cached rows (K_t, V_t)
                    e = acq(ain, ROW)
                    row = [e[i] for i in range(ROW)]
                    f_step(*row, qs, oacc, ml, pb, h0) if ACORES > 1 else f_step(*row, qs, oacc, ml, pb)
                    ain.release(ROW)
            if SHARED:                                              # this position's row, read back from the cache
                e = acq(ain, ROW)
                row = [e[i] for i in range(ROW)]
                f_stepn(*row, qs, oacc, ml, h0) if ACORES > 1 else f_stepn(*row, qs, oacc, ml)
                ain.release(ROW)
            else:
                f_stepn(kout, vout, qs, oacc, ml, h0) if ACORES > 1 else f_stepn(kout, vout, qs, oacc, ml)
            for hp in range_(N_OG):
                o = ogout.acquire(1)
                f_fin(oacc, ml, o, hp)
                ogout.release(1)
        return _attn

    def _bodies(self):
        """(core 0's body, make_body(c) for the rest). One shape of worker body per combination of
        knobs, not one with defaulted arguments: a family that does not block, or has no bias, must
        present IRON the exact function it presented before. The bias fifo is the worker's SECOND
        argument, before the drains."""
        _attn, NHL, SHARED, QKVB, RB = self._body_fns(), self.NHL, self.SHARED, self.QKVB, self.RB
        if SHARED:
            # no k / v work: every core, core 0 included, takes the extra cores' shape without the k / v scratch
            def make_attn_body(c):
                h0 = c * NHL
                if RB > 1:
                    def body(ain, ogout, qn, kn, cs, qs, oacc, ml, pb, f_meta, f_q, f_init, f_step, f_stepn, f_fin, f_stepb):
                        _attn(ain, None, ogout, qn, kn, cs, qs, None, None, None, oacc, ml, pb,
                              f_meta, f_q, None, None, f_init, f_step, f_stepn, f_fin, f_stepb, h0)
                else:
                    def body(ain, ogout, qn, kn, cs, qs, oacc, ml, pb, f_meta, f_q, f_init, f_step, f_stepn, f_fin):
                        _attn(ain, None, ogout, qn, kn, cs, qs, None, None, None, oacc, ml, pb,
                              f_meta, f_q, None, None, f_init, f_step, f_stepn, f_fin, None, h0)
                return body
            attn_body = make_attn_body(0)
        elif QKVB and RB > 1:
            def attn_body(ain, bias_in, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
                _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                      f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, 0, bias_in)

            def make_attn_body(c):
                h0 = c * NHL
                def body(ain, bias_in, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
                    _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                          f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, h0, bias_in)
                return body
        elif QKVB:
            def attn_body(ain, bias_in, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin):
                _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                      f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, None, 0, bias_in)

            def make_attn_body(c):
                h0 = c * NHL
                def body(ain, bias_in, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin):
                    _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                          f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, None, h0, bias_in)
                return body
        elif RB > 1:
            def attn_body(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
                _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                      f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, 0)

            def make_attn_body(c):
                h0 = c * NHL
                def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
                    _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                          f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, h0)
                return body
        else:
            def attn_body(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin):
                _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                      f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, None, 0)

            def make_attn_body(c):
                h0 = c * NHL
                def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin):
                    _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                          f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, None, h0)
                return body
        return attn_body, make_attn_body

    def workers(self, tiles):
        """`tiles[c]`: attention core c's tile."""
        attn_body, make_attn_body = self._bodies()
        s = self.sfx
        bhd, fcs, fq, fhd, brow, foacc, fml, pb_ty = self.bhd, self.fcs, self.fq, self.fhd, self.brow, self.foacc, self.fml, self.pb_ty
        RB, ACORES = self.RB, self.ACORES
        ws = []

        def abufs(c):
            n = s if c == 0 else f"{c}{s}"
            return [Buffer(bhd, name=f"qn{n}"), Buffer(bhd, name=f"kn{n}"), Buffer(fcs, name=f"cs{n}"),
                    Buffer(fq, name=f"qs{n}"), Buffer(fhd, name=f"tmp{n}"), Buffer(brow, name=f"kout{n}"),
                    Buffer(brow, name=f"vout{n}"), Buffer(foacc, name=f"oacc{n}"), Buffer(fml, name=f"ml{n}"),
                    Buffer(pb_ty, name=f"pb{n}")]

        if self.SHARED:
            def sbufs(c):
                n = s if c == 0 else f"{c}{s}"
                return [Buffer(bhd, name=f"qn{n}"), Buffer(bhd, name=f"kn{n}"), Buffer(fcs, name=f"cs{n}"),
                        Buffer(fq, name=f"qs{n}"), Buffer(foacc, name=f"oacc{n}"), Buffer(fml, name=f"ml{n}"),
                        Buffer(pb_ty, name=f"pb{n}")]
            sfns = [self.f_meta, self.f_q, self.f_init, self.f_step, self.f_stepn, self.f_fin] + ([self.f_stepb] if RB > 1 else [])
            for c in range(ACORES):
                ws.append(Worker(make_attn_body(c), fn_args=[self.of_ain.cons(), self.of_og[c].prod()] + sbufs(c) + sfns,
                                 tile=tiles[c], stack_size=0x1800))
        else:
            afns = [self.f_meta, self.f_q, self.f_k, self.f_v, self.f_init, self.f_step, self.f_stepn, self.f_fin] + (
                [self.f_stepb] if RB > 1 else [])
            bcons = (lambda: [self.of_abias.cons()]) if self.QKVB else (lambda: [])
            ws.append(Worker(attn_body, fn_args=[self.of_ain.cons()] + bcons() + [self.of_aout.prod(), self.of_og[0].prod()] + abufs(0) + afns,
                             tile=tiles[0], stack_size=0x1800))
            # The rest of the attention cores: same broadcast stream in, their own og out.
            for c in range(1, ACORES):
                ws.append(Worker(make_attn_body(c), fn_args=[self.of_ain.cons()] + bcons() + [self.of_og[c].prod()] + abufs(c) + afns,
                                 tile=tiles[c], stack_size=0x1800))
        return ws


class Bands:
    """Bytes of one 64-row band of each projection the sequence streams, at the role's weight format."""

    def __init__(self, BB_H, BB_Q, BB_F, BB_UG):
        self.BB_H, self.BB_Q, self.BB_F, self.BB_UG = BB_H, BB_Q, BB_F, BB_UG


def layer_sequence(G, L, stop, bands, a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod,
                   y_conss, ain_p, aout_c, og_cs, abias_p, *, totals=None, og_joined=False, y_at=None):
    """One dense layer's host sequence. `totals` widens the taps' buffer bounds to the declared argument
    sizes when several kinds share one image (a tap's total only bounds its offsets); `og_joined`: og_cs
    is the one joined drain of Attn.fifos(join=True); `y_at`: the FFN residual goes to this act offset
    instead of xres (gx.py's per-layer input runs after it; sandwich norms only)."""
    T = totals or {}
    P_B, A_B, C_B, K_B, R_B = (T.get("pool", L.POOL_BYTES), T.get("act", L.AD_BYTES), T.get("consts", L.CD_BYTES),
                               T.get("kv", L.KV_BYTES), T.get("ptab", L.PTAB_BYTES))
    HID, KVW, QW, N_CORES = G.HID, G.KVW, G.QW, G.N_CORES
    ELN, E_A = L.ELN, L.E_A
    SHARED, ACORES, NHL = G.KV_SHARED, G.ACORES, G.NHL
    BB_H, BB_Q, BB_F, BB_UG = bands.BB_H, bands.BB_Q, bands.BB_F, bands.BB_UG
    YB = BAND_ROWS * 4
    # 1. layer-entry norm: xn -> act
    tg_ln = TaskGroup()
    lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
    lni.fill(a_consts, tap=bt(C_B, L.CD_LNW, ELN), wait=True, group=tg_ln)
    lno.drain(a_act, tap=bt(A_B, L.AD_XN, ELN), wait=True, group=tg_ln)
    # 2. q | k | v: weights now, xn after the norm
    pw, py = Pipeline(3), Pipeline(3)
    for c in range(N_CORES):
        segs = qkv_segments(G, L, c, BB_H)
        for po, pn, _, _ in segs:
            pw.fill(w_prods[c], a_pool, bt(P_B, po, pn))
        for _, _, ao, an in segs:
            py.drain(y_conss[c], a_act, bt(A_B, ao, an))
    tg_ln.finish()                                            # xn is in DDR
    tg_x = TaskGroup()
    x_prod.fill(a_act, tap=bt(A_B, L.AD_XN, G.XN_ELEMS * ELEM), wait=True, group=tg_x)
    # DX_STOP=1 no longer early-returns: phase 3 (attention) runs for every stop
    # now, fed the placeholder 1-row window at stop=1, so the attention cores
    # complete their bodies and multi-run invocations stay clean (the early
    # return left every attention worker blocked mid-body).
    # 3. attention: meta + record now, q / k / v after the GEMVs, the window, the new row out
    pa_out, pa_in = Pipeline(3), Pipeline(3)
    if not SHARED:
        pa_out.drain(aout_c, a_kv, bt(K_B, L.KV_ROW, L.KV_ROW))      # [k' | v'] -> row pos (attnpos)
    if og_joined:
        pa_out.drain(og_cs, a_act, bt(A_B, L.AD_OG, ACORES * NHL * G.HD * 2))
    else:
        for c in range(ACORES):                                             # heads NHL*c ..
            pa_out.drain(og_cs[c], a_act, bt(A_B, L.AD_OG + c * NHL * G.HD * 2, NHL * G.HD * 2))
    pa_in.fill(ain_p, a_consts, bt(C_B, L.CD_META, E_A))            # [qn | kn]
    pa_in.fill(ain_p, a_ptab, bt(R_B, L.PTAB_ROW, L.PTAB_ROW))    # the position record (attnpos)
    py.finish(*y_conss)                                       # q, k, v are in DDR
    pa_in.fill(ain_p, a_act, bt(A_B, L.AD_Q, QW * 4))
    if not SHARED:
        pa_in.fill(ain_p, a_act, bt(A_B, L.AD_KVN, KVW * 4))
        pa_in.fill(ain_p, a_act, bt(A_B, L.AD_KVN + KVW * 4, KVW * 4))
    if abias_p is not None:                                   # one bias element per q / k / v element
        pa_in.fill(abias_p, a_consts, bt(C_B, L.CD_QB, QW * 2))
        pa_in.fill(abias_p, a_consts, bt(C_B, L.CD_KB, KVW * 2))
        pa_in.fill(abias_p, a_consts, bt(C_B, L.CD_VB, KVW * 2))
    pa_in.fill(ain_p, a_kv, bt(K_B, 0, L.KV_ROW))                   # the window: rows [0, nf) (attnpos)
    if SHARED:
        # this position's row, which the source layer wrote this token; at KV_ROW it takes the row drain's patch
        pa_in.fill(ain_p, a_kv, bt(K_B, L.KV_ROW, L.KV_ROW))
    # 4. o projection against og
    if stop >= 2:
        # at stop=1 the main cores return after q/k/v: nobody would consume
        # the o weights or drain the o results
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(P_B, L.POOL_O + c * G.O_PC * BB_Q, G.O_PC * BB_Q))
            py.drain(y_conss[c], a_act, bt(A_B, L.AD_OUT + c * G.O_PC * YB, G.O_PC * YB))
    pa_out.finish()                                           # og (and the new cache row) are in DDR
    if stop >= 2:
        # the og x-fill must PRECEDE the stop<=2 tail: the o gemv consumes
        # it, and the old early return starved it (the cores park in
        # state 8 for seconds waiting on an x fill that never arrives)
        x_prod.fill(a_act, tap=bt(A_B, L.AD_OG, G.OG_ELEMS * ELEM), wait=True, group=tg_x)
    if stop <= 2:
        pa_in.finish()
        py.finish()
        pw.finish()
        tg_x.finish()
        return
    # 5. residual + norms -> res, xm (sandwich: t = post_attn_norm(out) first, then res = x + t,
    #    xm = pre_ffn_norm(res); plain: res = x + out, xm = post_attn_norm(res))
    tg_ln2 = TaskGroup()
    if G.SANDWICH:
        py.finish()                                           # out is in DDR
        tg_t = TaskGroup()
        lni.fill(a_act, tap=bt(A_B, L.AD_OUT, HID * 4), wait=True, group=tg_t)
        lni.fill(a_consts, tap=bt(C_B, L.CD_POSTLN, ELN), wait=True, group=tg_t)
        lno.drain(a_act, tap=bt(A_B, L.AD_T, HID * 4), wait=True, group=tg_t)
        tg_t.finish()                                         # t is in DDR
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(C_B, L.CD_PREFFN, ELN), wait=True, group=tg_ln2)
        lni.fill(a_act, tap=bt(A_B, L.AD_T, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(A_B, L.AD_RES, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(A_B, L.AD_XM, ELN), wait=True, group=tg_ln2)
    else:
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(C_B, L.CD_POSTLN, ELN), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(A_B, L.AD_RES, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(A_B, L.AD_XM, ELN), wait=True, group=tg_ln2)
        py.finish()                                           # out is in DDR
        lni.fill(a_act, tap=bt(A_B, L.AD_OUT, HID * 4), wait=True, group=tg_ln2)
    tg_ln2.finish()                                           # res, xm are in DDR
    # 6. up | gate per band, silu -> h
    x_prod.fill(a_act, tap=bt(A_B, L.AD_XM, G.XM_ELEMS * ELEM), wait=True, group=tg_x)
    # The drains first (a core blocks on a full y fifo after two bands), then the per-band
    # fills interleaved across the cores so every core streams while the host paces the
    # 38 fills per core three at a time (Pipeline).
    for c in range(N_CORES):
        py.drain(y_conss[c], a_act, bt(A_B, L.AD_H + c * G.UP_PC * YB, G.UP_PC * YB))
    for j in range(G.UP_PC):
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(P_B, L.POOL_UP + (c * G.UP_PC + j) * BB_UG, BB_UG))
            pw.fill(w_prods[c], a_pool, bt(P_B, L.POOL_GATE + (c * G.UP_PC + j) * BB_UG, BB_UG))
    py.finish()                                               # h is in DDR
    if stop == 3:
        pw.finish()
        pa_in.finish()
        tg_x.finish()
        return
    # 7. down against h, then the output residual -> xres
    x_prod.fill(a_act, tap=bt(A_B, L.AD_H, G.H_ELEMS * ELEM), wait=True, group=tg_x)
    for c in range(N_CORES):
        pw.fill(w_prods[c], a_pool, bt(P_B, L.POOL_DOWN + c * G.DOWN_PC * BB_F, G.DOWN_PC * BB_F))
        py.drain(y_conss[c], a_act, bt(A_B, L.AD_OUT2 + c * G.DOWN_PC * YB, G.DOWN_PC * YB))
    tg_ln3 = TaskGroup()
    if G.SANDWICH:
        py.finish()                                           # out2 is in DDR
        tg_t2 = TaskGroup()
        lni.fill(a_act, tap=bt(A_B, L.AD_OUT2, HID * 4), wait=True, group=tg_t2)
        lni.fill(a_consts, tap=bt(C_B, L.CD_POSTFFN, ELN), wait=True, group=tg_t2)
        lno.drain(a_act, tap=bt(A_B, L.AD_T2, HID * 4), wait=True, group=tg_t2)
        tg_t2.finish()                                        # t2 is in DDR
        lni.fill(a_act, tap=bt(A_B, L.AD_RES, HID * 4), wait=True, group=tg_ln3)
        lni.fill(a_consts, tap=bt(C_B, L.CD_POSTFFN, ELN), wait=True, group=tg_ln3)   # unused w
        lni.fill(a_act, tap=bt(A_B, L.AD_T2, HID * 4), wait=True, group=tg_ln3)
        if y_at is None:
            lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln3)
        else:
            lno.drain(a_act, tap=bt(A_B, y_at, HID * 4), wait=True, group=tg_ln3)
    else:
        assert y_at is None
        lni.fill(a_act, tap=bt(A_B, L.AD_RES, HID * 4), wait=True, group=tg_ln3)
        lni.fill(a_consts, tap=bt(C_B, L.CD_POSTLN, ELN), wait=True, group=tg_ln3)
        lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln3)
        lno.drain(a_act, tap=bt(A_B, L.AD_JUNK, ELN), wait=True, group=tg_ln3)
        py.finish()                                           # out2 is in DDR
        lni.fill(a_act, tap=bt(A_B, L.AD_OUT2, HID * 4), wait=True, group=tg_ln3)
    tg_ln3.finish()
    pw.finish()
    pa_in.finish()
    tg_x.finish()
