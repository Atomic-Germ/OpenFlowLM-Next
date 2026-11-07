r"""The PRODUCTION dense attention worker, standalone, with synthetic inputs.

The gate that lets a new model SIZE be added without a model: it builds the
attention kernel the dense recipe ships (designs/dense/dx.py's worker, lifted
verbatim -- see probe_support.py) at one geometry, runs it against synthetic
q / k / v and a synthetic KV window, and compares the result with an fp64
oracle (compare.py).

Why it exists: a new size of a known family is a new attention GEOMETRY
(head dim, head count, kv-head count, rotary dim, the q/k norm and the gate),
and the catalogue refuses a geometry it has not seen run. The old way to see
one run was to convert the model, decode it and compare against the replica --
which needs the weights, the container, the engine and a person. This gate
needs none of that: the geometry and the numbers are synthetic.

What it does NOT cover: the projections' weights, the packing, the host's
window bookkeeping (which rows are valid is the driver's attnpos patch, not
the kernel) and the whole-layer integration. It covers the attention kernel's
own arithmetic at that geometry -- the thing the catalogue's tuple is about.

The build is the geometry, so the geometry comes from the environment:

    ATTN_PROBE_SPEC=open_kernels/recipes/specs/qwen3-0.6b.json \
    ATTN_PROBE_ROWS=17 \
      python open_kernels/build_design.py open_kernels/designs/attn/probe_dense.py <out>

The six buffers (the harness's opcode-3 ABI):

    meta   u8[(1 + NPTAB) * E_A]   [qn | kn] then the position record
    qg     f32[QW]                q (the dense recipe has no attention gate)
    kvn    f32[2 * KVW]           [k | v]
    cache  bf16[ROWS * 2 * KVW]   the window: [K_t | V_t] per row
    new    bf16[2 * KVW]          out: the new position's [k' | v'] rows
    og     bf16[NH * HD]          out: the attention output, all heads
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes import dense as D  # noqa: E402
from recipes.load import load_spec  # noqa: E402
from designs.attn import probe_support as P  # noqa: E402

SPEC_PATH = os.environ.get("ATTN_PROBE_SPEC", "")
ROWS = int(os.environ.get("ATTN_PROBE_ROWS", "17"))
if not SPEC_PATH:
    raise SystemExit("probe_dense: ATTN_PROBE_SPEC must name a kernel spec (recipes/specs/*.json)")
if ROWS < 1:
    raise SystemExit("probe_dense: ATTN_PROBE_ROWS must be at least 1")

SPEC = load_spec(Path(SPEC_PATH))
G = D.geometry(SPEC)
if G.QKVB:
    raise SystemExit("probe_dense: a family with a q/k/v bias is not covered by this gate")
if G.RB not in (1, 2, 4):
    raise SystemExit("probe_dense: only the single-row path and the block path (RB 2 or 4) are covered")

NH, KVH, HD = G.NH, G.KVH, G.HD
QW, KVW = G.QW, G.KVW
E_A = G.KVW * 2                      # one attention element: KVH heads of bf16
ACORES, NHL, RB = G.ACORES, G.NHL, G.RB
NPTAB, CSE = G.PTAB_ELEMS, G.PTAB_CS_ELEM
OGH = min(NHL, G.HPO)                # heads in one og element (attn.h's kOGH)
N_OG = NHL // OGH
FLAGS = D.attn_flags(G)
INC = include_dirs() + [str(HERE)]

BODY = P.worker(G, NPTAB, CSE, RB, ACORES, NHL, N_OG, KVW, range_)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def probe(meta: In, qg: In, kvn: In, cache: In, new: Out, og: Out, *, key: CompileTime[int]):
    def ty(n, dtype):
        return np.ndarray[(n,), np.dtype[dtype]]

    elem = ty(E_A, np.uint8)
    bhd = ty(HD, bfloat16)            # qn / kn: the norm weights
    brow = ty(KVW, bfloat16)          # a cache row half: KVH heads of bf16
    og_ty = ty(OGH * HD, bfloat16)    # attn_fin writes kOGH heads at a time
    fcs = ty(G.ROT, np.float32)       # [cos | sin]
    fhd = ty(HD, np.float32)
    fq = ty(2 * QW, bfloat16) if G.VEXP else ty(QW, np.float32)
    foacc = ty(NHL * HD, np.float32)  # this core's heads only
    fml = ty(2 * G.MLS, np.float32)   # [m | l]
    pb_ty = ty(8 if RB > 1 else 4, np.int32)
    i32 = np.int32

    def ef(sym, src, args, flags=FLAGS):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=INC, compile_flags=flags)

    pz = [elem] if NPTAB > 1 else []                       # the record's second element
    h0_arg = [i32] if ACORES > 1 else []                   # only a split needs the head offset
    f_meta = ef("attn_meta", HERE / "attn_meta.cc", [elem, elem] + pz + [bhd, bhd, fcs, pb_ty])
    f_q = ef("attn_q", HERE / "attn_q.cc", [elem, bhd, fcs, fq, i32])
    f_k = ef("attn_k", HERE / "attn_k.cc", [elem, bhd, fcs, fhd, brow, i32])
    f_v = ef("attn_v", HERE / "attn_v.cc", [elem, brow, i32])
    f_init = ef("attn_init", HERE / "attn_init.cc", [foacc, fml])
    f_step = ef("attn_step", HERE / "attn_step.cc", [elem, elem, fq, foacc, fml, pb_ty] + h0_arg)
    f_stepn = ef("attn_step_new", HERE / "attn_step_new.cc", [brow, brow, fq, foacc, fml] + h0_arg)
    f_stepb = (ef("attn_stepb", HERE / "attn_stepb.cc", [elem] * (2 * RB) + [fq, foacc, fml, pb_ty] + h0_arg)
               if RB > 1 else None)
    f_fin = ef("attn_fin_ng", HERE / "attn_fin_ng.cc", [foacc, fml, og_ty, i32])

    inp = ObjectFifo(elem, name="ain", depth=max(4, 2 * RB + 2, 1 + NPTAB + 1))
    aout = ObjectFifo(brow, name="aout", depth=2)
    ogs = [ObjectFifo(og_ty, name=f"og{c}", depth=2) for c in range(ACORES)]

    def attn_body(ain, aout_, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                  f_meta_, f_q_, f_k_, f_v_, f_init_, f_step_, f_stepn_, f_fin_, f_stepb_):
        BODY(ain, aout_, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
             f_meta_, f_q_, f_k_, f_v_, f_init_, f_step_, f_stepn_, f_fin_, f_stepb_, 0)

    def make_attn_body(c):
        h0 = c * NHL

        def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                 f_meta_, f_q_, f_k_, f_v_, f_init_, f_step_, f_stepn_, f_fin_, f_stepb_):
            BODY(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                 f_meta_, f_q_, f_k_, f_v_, f_init_, f_step_, f_stepn_, f_fin_, f_stepb_, h0)
        return body

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s}"), Buffer(bhd, name=f"kn{s}"), Buffer(fcs, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(fhd, name=f"tmp{s}"), Buffer(brow, name=f"kout{s}"),
                Buffer(brow, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]

    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin] + ([f_stepb] if RB > 1 else [])
    workers = [Worker(attn_body, fn_args=[inp.cons(), aout.prod(), ogs[0].prod()] + abufs(0) + afns,
                      tile=Tile(0, 2), stack_size=0x1800)]
    for c in range(1, ACORES):
        workers.append(Worker(make_attn_body(c), fn_args=[inp.cons(), ogs[c].prod()] + abufs(c) + afns,
                              tile=Tile(c, 2), stack_size=0x1800))

    def sequence(a_meta, a_qg, a_kvn, a_cache, a_new, a_og, ip, outs):
        def tap(n, o, l):
            return TensorAccessPattern((1, n), o, [1, 1, 1, l], [0, 0, 0, 1])

        pipe = Pipeline(3)
        # out: the new row [k' | v'], then each core's own og element
        pipe.drain(outs[0], a_new, tap(2 * KVW, 0, KVW))
        pipe.drain(outs[0], a_new, tap(2 * KVW, KVW, KVW))
        for c in range(ACORES):
            pipe.drain(outs[1 + c], a_og, tap(NH * HD, c * NHL * HD, NHL * HD))
        # in: [qn | kn], the position record, q, k, v, the window
        pipe.fill(ip, a_meta, tap((1 + NPTAB) * E_A, 0, E_A))
        pipe.fill(ip, a_meta, tap((1 + NPTAB) * E_A, E_A, NPTAB * E_A))
        pipe.fill(ip, a_qg, tap(QW, 0, QW))
        pipe.fill(ip, a_kvn, tap(2 * KVW, 0, KVW))
        pipe.fill(ip, a_kvn, tap(2 * KVW, KVW, KVW))
        pipe.fill(ip, a_cache, tap(ROWS * 2 * KVW, 0, ROWS * 2 * KVW))
        pipe.finish()

    rt = Runtime(sequence, [ty((1 + NPTAB) * E_A, np.uint8), ty(QW, np.float32), ty(2 * KVW, np.float32),
                            ty(ROWS * 2 * KVW, bfloat16), ty(2 * KVW, bfloat16), ty(NH * HD, bfloat16),
                            inp.prod(), [aout.cons()] + [o.cons() for o in ogs]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = probe
_sources = [*HERE.glob("*.cc"), HERE / "attn.h", HERE / "probe_support.py", HERE / "probe_dense.py",
            P.DX, ROOT / "include" / "vecmath.h"]
_raw = b"".join(p.read_bytes() for p in sorted(_sources)) + repr((G, ROWS, NPTAB, CSE, OGH, N_OG)).encode()
SPECIALIZE = {"key": int(hashlib.sha256(_raw).hexdigest()[:8], 16)}
