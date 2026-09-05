r"""moe_experts: a whole MoE block (8 routed experts + shared expert + combine)
as ONE dispatch.

    acc  = sum_e w[e] * down_e( bf16( silu(gate_e @ xm) * (up_e @ xm) ) )
    out  = xres + acc + sigmoid(xm . sgw) * down_s( bf16( silu(gate_s @ xm) * (up_s @ xm) ) )

Phase 2 steps 1 + 1b of .claude/plans/open-kernels-phase2-moe-first.md:
replaces the 45 host-driven dispatches (5 per routed expert over 4 xclbin
contexts, 5 for the shared expert + combine) of moe_chain / decode_chain with
one. Weights are the same pool-order chunks make_27b.py slices today,
concatenated per expert `[up 4 stripes | gate 4 stripes | down 16 bands]` =
1,966,080 B x 8, then the shared expert `[share_up | share_gate | share_down]`
(the same 3 x 655,360 B, standard RS=2 layout) as a 9th. (Step 2a streams the
same bytes from the resident layer pool: the driver's `moeroute` rewrites each
fill's offset, keeping its position inside the stripe.)

Cores (one per column, Tile(c, 2)), each with exactly its 2 input DMA channels
(w from the shim, h from the memtile) and <= 2 outputs (h part, acc). Phase 2
item 5 balance: ALL 8 cores do up/gate (64 rows each) as well as down:
  all 8 : per routed expert, the 64-row half c%2 of up stripe c//2 then of gate
          stripe c//2 -- the stripe is an RS=4 band (quarter = chunk%4, k-tile
          = chunk/4), so one half is the chunk pairs {4kt + 2(c%2), +1}: a
          strided DMA tap (8 x 10240 B, stride 20480) that the RS=2 band law
          (half = chunk%2, k-tile = chunk/2) consumes as a plain 64-row band.
          For the shared expert (RS=2 layout) it is simply band c. Against xm
          -> h_c = bf16(silu(g)*u), 64 rows.
  pairs : the odd core hands its 64 h rows to its even neighbour through
          shared L1 (an AIE2 core reads its west neighbour's memory); the even
          core emits hp = [own 64 | neighbour 64] = 128 rows of stripe c//2,
          joined on a memtile (a memtile has 6 DMA inputs, so 4 producers)
          into h[512] and broadcast to all 8 cores.
  all 8 : per expert, its 256 rows of the down projection (two 8-chunk RS=4
          bands / four 4-chunk RS=2 bands) against h; routed: acc += w[e]*y in
          the output element; shared: out = xres + acc + gate*y, drained once.
The first element of every core's w stream is the header
[xm | router output | sgw | xres] (see moe_hdr.cc).

Activations go through gemv_q4's int16 block tables (gemv_q4_prep_k2048 on the
header element for xm, gemv_q4_prep_k512 on each h element) -- phase 2 item 5.

Args: wexp u8[9 * 1966080], hdr u8[20480], out f32[2048].
Build (WSL): python build_design.py designs/moe_experts/moe_experts.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

NE = 8                                # routed experts
NX = NE + 1                           # + the shared expert, streamed the same way
HID = 2048
FF = 512
TILE = 5120
PER_CALL = 4
CALL_BYTES = PER_CALL * TILE          # one w element (and the header)
STRIPE = 32 * TILE                    # 128 rows x 2048: one up/gate band (RS=4) or two (RS=2)
HALF = 16 * TILE                      # 64 rows x 2048 = 4 elements: one core's share of a stripe
PAIR = 2 * TILE                       # the two chunks of one half at one k-tile
DOWN_BAND = 8 * TILE                  # 128 rows x 512 (RS=4) or two 64-row bands (RS=2)
UP_BYTES = 4 * STRIPE                 # 655360
EXPERT_BYTES = 3 * UP_BYTES           # up | gate | down
N_CORES = 8
N_PAIRS = N_CORES // 2                # h parts joined on the memtile
DOWN_PER_CORE = 2                     # 16 x 128-row down bands / 8 cores
HDR_BYTES = 20480
ROWS = 64                             # up/gate rows per core


def tap(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def half_tap(total: int, off: int) -> TensorAccessPattern:
    """One 64-row half of an RS=4 stripe: chunk pairs at every k-tile (8 x 10240 B, stride 20480).
    Three real dims (8 k-tiles x 4 x 2560 B): the shim BD's highest dim is a repeat count, its
    length covers only the lowest three, and the innermost wrap is 10 bits (< 4096 B)."""
    return TensorAccessPattern((1, total), off, [1, 8, 4, PAIR // 4], [0, 2 * PAIR, PAIR // 4, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def moe_experts(wexp: In, hdr: In, out: Out, *, srchash: CompileTime[int] = 0):
    w_ty = np.ndarray[(NX * EXPERT_BYTES,), np.dtype[np.uint8]]
    elem_ty = np.ndarray[(CALL_BYTES,), np.dtype[np.uint8]]
    h_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    hp_ty = np.ndarray[(2 * ROWS,), np.dtype[bfloat16]]        # a pair's 128 rows of h
    nb_ty = hp_ty                                              # the odd core's 64 rows (in a 128 element: one silu signature)
    r_ty = np.ndarray[(32,), np.dtype[np.float32]]     # router floats 256..287; [0] = shared gate
    band_ty = np.ndarray[(128,), np.dtype[np.float32]]
    accp_ty = np.ndarray[(DOWN_PER_CORE * 128,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    tabx_ty = np.ndarray[(2 * HID + HID // 4,), np.dtype[np.uint8]]   # gemv_q4_tab_bytes(2048): xm
    tabh_ty = np.ndarray[(2 * FF + FF // 4,), np.dtype[np.uint8]]     # gemv_q4_tab_bytes(512): h

    inc = include_dirs() + [str(GEMV)]
    # up/gate: 64-row bands of 16 chunks, RS=2 band law (4 groups of 4) -- the
    # routed halves through the strided tap, the shared bands as they lie.
    # down: routed 8-chunk RS=4 bands (2 groups); shared RS=2 wrappers with a
    # runtime group + output offset (two 64-row bands per 128-float buffer).
    k_up = [ExternalFunction(f"gemv_q4_p4b16r2_k{i}", source_file=str(GEMV / f"gemv_q4_p4b16r2_k{i}.cc"),
                             arg_types=[elem_ty, tabx_ty, band_ty], include_dirs=inc) for i in range(4)]
    k_dn = [ExternalFunction(f"gemv_q4_p4b8r4_k{i}", source_file=str(GEMV / f"gemv_q4_p4b8r4_k{i}.cc"),
                             arg_types=[elem_ty, tabh_ty, band_ty], include_dirs=inc) for i in range(2)]
    r2h = ExternalFunction("gemv_q4_r2h", source_file=str(HERE / "gemv_q4_r2h.cc"),
                           arg_types=[elem_ty, tabh_ty, band_ty, np.int32, np.int32], include_dirs=inc)
    # activation prep (gemv_q4.h): xm straight from the header element, h from its fifo element
    prepx = ExternalFunction("gemv_q4_prep_k2048", source_file=str(GEMV / "gemv_q4_prep_k2048.cc"),
                             arg_types=[elem_ty, tabx_ty], include_dirs=inc)
    preph = ExternalFunction("gemv_q4_prep_k512", source_file=str(GEMV / "gemv_q4_prep_k512.cc"),
                             arg_types=[h_ty, tabh_ty], include_dirs=inc)
    hdrf = ExternalFunction("moe_hdr", source_file=str(HERE / "moe_hdr.cc"),
                            arg_types=[elem_ty, r_ty, accp_ty, np.int32], include_dirs=inc)
    silu = ExternalFunction("moe_silu", source_file=str(HERE / "moe_silu.cc"),
                            arg_types=[band_ty, band_ty, nb_ty], include_dirs=inc)
    catf = ExternalFunction("moe_cat", source_file=str(HERE / "moe_cat.cc"),
                            arg_types=[nb_ty, hp_ty], include_dirs=inc)
    accf = ExternalFunction("moe_acc", source_file=str(HERE / "moe_acc.cc"),
                            arg_types=[band_ty, band_ty, r_ty, accp_ty, np.int32], include_dirs=inc)
    finf = ExternalFunction("moe_fin", source_file=str(HERE / "moe_fin.cc"),
                            arg_types=[band_ty, band_ty, r_ty, accp_ty, accp_ty], include_dirs=inc)

    of_w = [ObjectFifo(elem_ty, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_h = ObjectFifo(h_ty, name="h", depth=2)
    of_hp = of_h.prod().join([p * 2 * ROWS for p in range(N_PAIRS)],
                             obj_types=[hp_ty] * N_PAIRS, names=[f"hp{p}" for p in range(N_PAIRS)],
                             depths=[2] * N_PAIRS)
    of_nb = [ObjectFifo(nb_ty, name=f"nb{p}", depth=2) for p in range(N_PAIRS)]   # odd core -> even core, shared L1
    of_acc = [ObjectFifo(accp_ty, name=f"acc{c}", depth=1) for c in range(N_CORES)]

    def up_gate(win, tabx, ub, gb, kup):
        for dst in (ub, gb):
            for fn in kup:
                we = win.acquire(1)
                fn(we, tabx, dst)
                win.release(1)

    def routed_down(win, tabh, y0, y1, kdn):
        for yb in (y0, y1):
            for fn in kdn:
                we = win.acquire(1)
                fn(we, tabh, yb)
                win.release(1)

    def shared_down(win, tabh, y0, y1):
        for j in range(4):                       # 64-row bands 4c..4c+3 -> [y0 | y1]
            we = win.acquire(1)
            r2h(we, tabh, y0 if j < 2 else y1, 0, 64 * (j % 2))
            win.release(1)

    def body_even(win, hout, nbin, hin, aout, tabx, tabh, rb, xr, ub, gb, y0, y1, c,
                  fhdr, fprepx, fpreph, fsilu, fcat, facc, ffin, fr2h, *ks):
        kup, kdn = ks[:4], ks[4:]
        we = win.acquire(1)
        fhdr(we, rb, xr, c)
        fprepx(we, tabx)                          # xm -> GEMV table
        win.release(1)
        ae = aout.acquire(1)
        for e in range_(NE):
            up_gate(win, tabx, ub, gb, kup)
            he = hout.acquire(1)
            fsilu(gb, ub, he)                     # rows 0..63 of the pair's 128
            nb = nbin.acquire(1)
            fcat(nb, he)                          # rows 64..127 from the odd neighbour
            nbin.release(1)
            hout.release(1)
            hh = hin.acquire(1)
            fpreph(hh, tabh)                      # h -> GEMV table
            routed_down(win, tabh, y0, y1, kdn)
            facc(y0, y1, rb, ae, e)
            hin.release(1)
        # the shared expert: band c of up then gate, four RS=2 down bands, combine
        up_gate(win, tabx, ub, gb, kup)
        he = hout.acquire(1)
        fsilu(gb, ub, he)
        nb = nbin.acquire(1)
        fcat(nb, he)
        nbin.release(1)
        hout.release(1)
        hh = hin.acquire(1)
        fpreph(hh, tabh)
        shared_down(win, tabh, y0, y1)
        ffin(y0, y1, rb, xr, ae)
        hin.release(1)
        aout.release(1)

    def body_odd(win, hout, hin, aout, tabx, tabh, rb, xr, ub, gb, y0, y1, c,
                 fhdr, fprepx, fpreph, fsilu, facc, ffin, fr2h, *ks):
        kup, kdn = ks[:4], ks[4:]
        we = win.acquire(1)
        fhdr(we, rb, xr, c)
        fprepx(we, tabx)
        win.release(1)
        ae = aout.acquire(1)
        for e in range_(NE):
            up_gate(win, tabx, ub, gb, kup)
            he = hout.acquire(1)
            fsilu(gb, ub, he)                     # the odd core's 64 rows, to the even neighbour
            hout.release(1)
            hh = hin.acquire(1)
            fpreph(hh, tabh)
            routed_down(win, tabh, y0, y1, kdn)
            facc(y0, y1, rb, ae, e)
            hin.release(1)
        up_gate(win, tabx, ub, gb, kup)
        he = hout.acquire(1)
        fsilu(gb, ub, he)
        hout.release(1)
        hh = hin.acquire(1)
        fpreph(hh, tabh)
        shared_down(win, tabh, y0, y1)
        ffin(y0, y1, rb, xr, ae)
        hin.release(1)
        aout.release(1)

    workers = []
    for c in range(N_CORES):
        p = c // 2
        tabx = Buffer(tabx_ty, name=f"tabx{c}")
        tabh = Buffer(tabh_ty, name=f"tabh{c}")
        rb = Buffer(r_ty, name=f"r{c}")
        xr = Buffer(accp_ty, name=f"xr{c}")
        y0 = Buffer(band_ty, name=f"y0_{c}")
        y1 = Buffer(band_ty, name=f"y1_{c}")
        ub = Buffer(band_ty, name=f"u{c}")
        gb = Buffer(band_ty, name=f"g{c}")
        if c % 2 == 0:
            workers.append(Worker(body_even,
                                  fn_args=[of_w[c].cons(), of_hp[p].prod(), of_nb[p].cons(), of_h.cons(), of_acc[c].prod(),
                                           tabx, tabh, rb, xr, ub, gb, y0, y1, c,
                                           hdrf, prepx, preph, silu, catf, accf, finf, r2h,
                                           *k_up, *k_dn],
                                  tile=Tile(c, 2), stack_size=0x1800))
        else:
            workers.append(Worker(body_odd,
                                  fn_args=[of_w[c].cons(), of_nb[p].prod(), of_h.cons(), of_acc[c].prod(),
                                           tabx, tabh, rb, xr, ub, gb, y0, y1, c,
                                           hdrf, prepx, preph, silu, accf, finf, r2h,
                                           *k_up, *k_dn],
                                  tile=Tile(c, 2), stack_size=0x1800))

    W_TOTAL = NX * EXPERT_BYTES
    acc_taps = [tap(HID, c * DOWN_PER_CORE * 128, DOWN_PER_CORE * 128) for c in range(N_CORES)]

    def sequence(a_w, a_hdr, c_out, w_prods, acc_conss):
        tg_end = TaskGroup()
        for c in range(N_CORES):
            acc_conss[c].drain(c_out, tap=acc_taps[c], wait=True, group=tg_end)
        pipe = Pipeline(3)
        for c in range(N_CORES):
            pipe.fill(w_prods[c], a_hdr, tap(HDR_BYTES, 0, HDR_BYTES))
        for e in range(NX):
            base = e * EXPERT_BYTES
            for c in range(N_CORES):
                if e < NE:
                    # half c%2 of up stripe c//2, then of gate stripe c//2 (strided)
                    pipe.fill(w_prods[c], a_w, half_tap(W_TOTAL, base + (c // 2) * STRIPE + (c % 2) * PAIR))
                    pipe.fill(w_prods[c], a_w, half_tap(W_TOTAL, base + UP_BYTES + (c // 2) * STRIPE + (c % 2) * PAIR))
                else:
                    # the shared expert's RS=2 layout: band c of up, of gate (contiguous)
                    pipe.fill(w_prods[c], a_w, tap(W_TOTAL, base + c * HALF, HALF))
                    pipe.fill(w_prods[c], a_w, tap(W_TOTAL, base + UP_BYTES + c * HALF, HALF))
                pipe.fill(w_prods[c], a_w, tap(W_TOTAL, base + 2 * UP_BYTES + c * DOWN_PER_CORE * DOWN_BAND,
                                                DOWN_PER_CORE * DOWN_BAND))
        pipe.finish()
        tg_end.finish()

    rt = Runtime(sequence, [w_ty, np.ndarray[(HDR_BYTES,), np.dtype[np.uint8]], acc_ty,
                            [f.prod() for f in of_w], [f.cons() for f in of_acc]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = moe_experts
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + [(GEMV / "gemv_q4.h").read_bytes(),
                                                                       (HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
