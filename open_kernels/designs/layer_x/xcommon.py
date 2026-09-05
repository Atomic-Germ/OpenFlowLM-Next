r"""Shared pieces of the whole-layer designs lx / ax: the main cores' GEMV entry
points, the MoE block and the DeltaNet step on those cores (kernels, the core
program fragments, the host-sequence fragments), the norm + router helper core,
and the DMA tap helpers. See lx.py for the design as a whole.

Program memory (16 KB per core) shapes everything here: the main core's IRON
program alone was 10 KB with one kernel call site per stage, so the kernels
take ONE scratch buffer each (`ms` for the MoE, `ds` for DeltaNet, fixed
offsets inside), the routed and shared experts share one 9-iteration loop
(the down band law and acc/combine are chosen INSIDE the kernels from the slot
index), and every GEMV shape is one runtime-parameterised entry point.

Main-core streams (all layer types):
  w (10 KB elements from the shim): weights, the MoE header, experts, S slices, DeltaNet records
  x (4 KB elements, broadcast): xn, og (2 elements), xm, the expert hidden h (f32[512])
  y (256 B elements to the shim): band results, S' half rows, o, hidden parts, the block output
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

from layout import POOL_BYTES, POOL_DOWN, POOL_SHARE_DOWN, POOL_SHARE_GATE, POOL_SHARE_UP

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"

NE, NX = 8, 9                         # routed experts, + the shared one
HID, FF = 2048, 512
TILE = 5120
PER_CALL = 2
CALL_BYTES = PER_CALL * TILE          # 10240: one w element
STRIPE = 32 * TILE                    # 128 rows x 2048 (RS=4 band)
HALF = 16 * TILE                      # 64 rows x 2048 = 8 elements
PAIR = 2 * TILE                       # the two chunks of one half at one k-tile = 1 element
DOWN_BAND = 8 * TILE                  # 128 rows x 512
UP_BYTES = 4 * STRIPE                 # 655360 per projection
N_CORES = 8
DOWN_PER_CORE = 2
BAND_ROWS = 64
BAND16, BAND32 = 16 * TILE, 32 * TILE  # K=2048 / K=4096 band bytes
N_HDR = 3
OS = ["-Os"]                          # main-core kernels: size over speed (the GEMV is DMA-bound)

# scratch layouts (floats) -- keep in sync with moe_x.h / dnx.h
MS_FLOATS = 928
DS_FLOATS = 1280
TAB_BYTES = 2 * 4096 + 4096 // 4      # gemv_q4_tab_bytes(4096); K=2048 uses a prefix, h's table sits at +4608


def bt(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def half_tap(off: int) -> TensorAccessPattern:
    """One 64-row half of an RS=4 stripe in the pool: chunk pairs at every k-tile (8 x 10240 B,
    stride 20480), as three real DMA dims (the BD's highest dim is a repeat count, its length
    covers only the lowest three, the innermost wrap is < 4096 B)."""
    return TensorAccessPattern((1, POOL_BYTES), off, [1, 8, 4, PAIR // 4], [0, 2 * PAIR, PAIR // 4, 1])


def types():
    t = {}
    t["elem"] = np.ndarray[(CALL_BYTES,), np.dtype[np.uint8]]
    t["x"] = np.ndarray[(HID,), np.dtype[bfloat16]]            # one 4 KB act element (bf16 view)
    t["y"] = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]    # one y element
    t["tab"] = np.ndarray[(TAB_BYTES,), np.dtype[np.uint8]]
    t["ms"] = np.ndarray[(MS_FLOATS,), np.dtype[np.float32]]
    t["ds"] = np.ndarray[(DS_FLOATS,), np.dtype[np.float32]]
    return t


def kernels(inc, t):
    e, x, y, tab, ms, ds = t["elem"], t["x"], t["y"], t["tab"], t["ms"], t["ds"]
    i32 = np.int32

    def ef(sym, args):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc, compile_flags=OS)

    k = {}
    # GEMVs (runtime group / band law): projections into a y element; MoE up/gate and down into ms
    k["gy"] = ef("gemv_q4_gy", [e, tab, y, i32, i32, i32])          # (t, tab, ye, group, per_band, rs)
    k["gup"] = ef("gemv_q4_gup", [e, tab, ms, i32, i32])            # (t, tab, ms, group, band)  u | g
    k["gdown"] = ef("gemv_q4_gdown", [e, tab, ms, i32, i32])        # (t, tab, ms, j, slot)      routed / shared law
    # activation tables
    k["prep2048"] = ExternalFunction("gemv_q4_prep_k2048", source_file=str(GEMV / "gemv_q4_prep_k2048.cc"),
                                     arg_types=[x, tab], include_dirs=inc, compile_flags=OS)
    k["prep4096a"] = ef("gemv_q4_prep_k4096_b0n64", [x, tab])
    k["prep4096b"] = ef("gemv_q4_prep_k4096_b64n64", [x, tab])
    k["prepf"] = ef("gemv_q4_prep_h", [x, tab])                     # h (f32[512] in the element) -> tab + 4608
    # MoE
    k["hdr"] = ef("moe_hdr2", [e, x, ms, i32])
    k["silu"] = ef("moe_silu32", [ms, y])
    k["accfin"] = ef("moe_accfin", [ms, i32])
    k["out"] = ef("moe_out", [ms, y, i32])
    # DeltaNet
    k["vcopy"] = ef("dnx_vcopy", [e, ds])
    k["p1"] = ef("dnx_pass1", [e, ds, i32])
    k["delta"] = ef("dnx_delta", [ds])
    k["row"] = ef("dnx_row", [e, ds, y, i32, i32])
    k["ofin"] = ef("dnx_ofin", [ds, y, i32])
    return k


KNAMES = ("gy", "gup", "gdown", "prep2048", "prep4096a", "prep4096b", "prepf", "hdr", "silu", "accfin", "out",
          "vcopy", "p1", "delta", "row", "ofin")
BNAMES = ("tab", "ms", "ds")


def core_buffers(t, c):
    return dict(tab=Buffer(t["tab"], name=f"tab{c}"), ms=Buffer(t["ms"], name=f"ms{c}"), ds=Buffer(t["ds"], name=f"ds{c}"))


def worker_args(B, K):
    return [*[B[n] for n in BNAMES], *[K[n] for n in KNAMES]]


def unpack_args(args):
    B = dict(zip(BNAMES, args[:len(BNAMES)]))
    K = dict(zip(KNAMES, args[len(BNAMES):]))
    return B, K


def gemv_bands(win, yout, tab, gy, nbands, ngroups, per_band, rs):
    """nbands bands of ngroups elements each against the table, one y element per band."""
    for _ in range_(nbands):
        ye = yout.acquire(1)
        for g in range_(ngroups):
            we = win.acquire(1)
            gy(we, tab, ye, g, per_band, rs)
            win.release(1)
        yout.release(1)


# ---- the MoE block on one main core
def moe_body(win, ain, yout, B, K):
    tab, ms = B["tab"], B["ms"]
    xm = ain.acquire(1)
    K["prep2048"](xm, tab)
    for mode in range_(N_HDR):
        we = win.acquire(1)
        K["hdr"](we, xm, ms, mode)
        win.release(1)
    ain.release(1)
    for e in range_(NX):                          # 8 routed slots, then the shared expert
        for b in range_(2):                       # u then g, 64 rows each, 8 elements per band
            for g in range_(8):
                we = win.acquire(1)
                K["gup"](we, tab, ms, g, b)
                win.release(1)
        he = yout.acquire(1)
        K["silu"](ms, he)                         # this core's 64 rows of the hidden, f32, to DDR
        yout.release(1)
        hh = ain.acquire(1)
        K["prepf"](hh, tab)                       # the whole h back, -> its table (tab + 4608)
        ain.release(1)
        for j in range_(8):                       # the core's 256 down rows: 8 elements either way
            we = win.acquire(1)
            K["gdown"](we, tab, ms, j, e)
            win.release(1)
        K["accfin"](ms, e)                        # routed: acc += w[e] y; shared: out = xres + acc + gate y
    for j in range_(4):
        ye = yout.acquire(1)
        K["out"](ms, ye, j)
        yout.release(1)


def moe_sequence(pipe_w, pipe_x, pipe_y, a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                 A_BYTES, C_BYTES, A_XM, A_ROUT, A_RES, A_HP, C_SGW):
    """Host sequence of the MoE block (one instruction-stream part). Routed slot j's fills carry
    placeholder pool offsets (expert j); moeroute2 rewrites them from the router output."""
    pipe_x.fill(x_prod, a_act, bt(A_BYTES, A_XM, 4096))
    for c in range(N_CORES):
        pipe_w.fill(w_prods[c], a_act, bt(A_BYTES, A_ROUT, CALL_BYTES))
        pipe_w.fill(w_prods[c], a_consts, bt(C_BYTES, C_SGW, CALL_BYTES))
        pipe_w.fill(w_prods[c], a_act, bt(A_BYTES, A_RES + c * 1024, CALL_BYTES))
    for e in range(NX):
        for c in range(N_CORES):
            if e < NE:
                up = (8 * e + 2 * (c // 2)) * STRIPE + (c % 2) * PAIR
                pipe_w.fill(w_prods[c], a_pool, half_tap(up))
                pipe_w.fill(w_prods[c], a_pool, half_tap(up + STRIPE))
            else:
                pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_SHARE_UP + c * HALF, HALF))
                pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_SHARE_GATE + c * HALF, HALF))
            pipe_y.drain(y_conss[c], a_act, bt(A_BYTES, A_HP + c * 256, 256))
        pipe_y.finish(*y_conss)                           # the 8 hidden parts are in DDR
        pipe_x.fill(x_prod, a_act, bt(A_BYTES, A_HP, 4096))
        for c in range(N_CORES):
            if e < NE:
                pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_DOWN + e * UP_BYTES + c * DOWN_PER_CORE * DOWN_BAND,
                                                   DOWN_PER_CORE * DOWN_BAND))
            else:
                pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_SHARE_DOWN + c * DOWN_PER_CORE * DOWN_BAND,
                                                   DOWN_PER_CORE * DOWN_BAND))
    for c in range(N_CORES):
        pipe_y.drain(y_conss[c], c_xres, bt(HID, c * 256, 256))   # the block output = the new residual
    pipe_w.finish()
    pipe_x.finish()
    pipe_y.finish()


# ---- DeltaNet on the main cores (dnx.h): S slices ride the w stream, S' rows leave through y
DN_ROWS, DN_SLICES, DN_HEADS_PC = 20, 7, 4


def dn_body(win, yout, B, K):
    """This core's 4 heads: the record (copied out of its element: release() frees the OLDEST held
    element), 7 slices (pass 1), delta, 7 slices x 40 half rows (pass 2, into y elements), o."""
    ds = B["ds"]
    for _ in range_(DN_HEADS_PC):
        re_ = win.acquire(1)
        K["vcopy"](re_, ds)
        win.release(1)
        for blk in range_(DN_SLICES):
            se = win.acquire(1)
            K["p1"](se, ds, blk)
            win.release(1)
        K["delta"](ds)
        for blk in range_(DN_SLICES):
            se = win.acquire(1)
            for j in range_(2 * DN_ROWS):
                ye = yout.acquire(1)
                K["row"](se, ds, ye, blk, j)
                yout.release(1)
            win.release(1)
        for hf in range_(2):
            ye = yout.acquire(1)
            K["ofin"](ds, ye, hf)
            yout.release(1)


def dn_sequence(pipe_w, pipe_y, a_state, a_act, w_prods, y_conss, A_BYTES, A_VEC, A_O, STATE_BYTES, STATE_S_OFF,
                S_HEAD_BYTES):
    """Per core, per head: the record, S twice (pass 1, pass 2), S' back in place, o -> act[A_O]."""
    for c in range(N_CORES):
        for h in range(DN_HEADS_PC):
            hd = c * DN_HEADS_PC + h
            pipe_w.fill(w_prods[c], a_act, bt(A_BYTES, A_VEC + hd * 2048, CALL_BYTES))
            pipe_w.fill(w_prods[c], a_state, bt(STATE_BYTES, STATE_S_OFF + hd * S_HEAD_BYTES, S_HEAD_BYTES))
            pipe_y.drain(y_conss[c], a_state, bt(STATE_BYTES, STATE_S_OFF + hd * S_HEAD_BYTES, S_HEAD_BYTES))
            pipe_y.drain(y_conss[c], a_act, bt(A_BYTES, A_O + hd * 512, 512))
            pipe_w.fill(w_prods[c], a_state, bt(STATE_BYTES, STATE_S_OFF + hd * S_HEAD_BYTES, S_HEAD_BYTES))


# ---- the norm + router helper core (both layer types): ln_nr -> ln(+residual) -> router
LN = HERE.parent / "ln"
LINL = HERE.parent / "lin_layer"
RT = HERE.parent / "router"
W_ELEMS = HID * 256 * 2 // 4096       # router W as 256 elements of 4 KB


def ln_types():
    return dict(u8_4k=np.ndarray[(4096,), np.dtype[np.uint8]], xb=np.ndarray[(HID,), np.dtype[bfloat16]],
                racc=np.ndarray[(256,), np.dtype[np.float32]])


def ln_kernels(inc, t):
    u = t["u8_4k"]
    k = {}
    k["ln_nr"] = ExternalFunction("ln_nr", source_file=str(LINL / "ln_nr.cc"), arg_types=[u] * 4, include_dirs=inc)
    k["ln"] = ExternalFunction("ln_fn", source_file=str(LN / "ln.cc"), arg_types=[u] * 8, include_dirs=inc)
    k["rcopy"] = ExternalFunction("router_copy_x", source_file=str(RT / "router_copy.cc"), arg_types=[u, t["xb"]], include_dirs=inc)
    k["racc"] = ExternalFunction("router_acc", source_file=str(RT / "router.cc"), arg_types=[u, t["xb"], t["racc"], np.int32], include_dirs=inc)
    k["rfin"] = ExternalFunction("router_fin", source_file=str(RT / "router_fin.cc"), arg_types=[t["racc"], u], include_dirs=inc)
    return k


def ln_router_body(ain, aout, xs, acc, f_nr, f_ln, f_rc, f_ra, f_rf):
    """in: [x0 x1 w] -> out [xn];  in: [x0 x1 w a0 a1] -> out [y0 y1 xm];  in: W x256 -> out [rout]"""
    e = ain.acquire(3)
    o = aout.acquire(1)
    f_nr(e[0], e[1], e[2], o)
    aout.release(1)
    ain.release(3)
    e = ain.acquire(5)
    oo = aout.acquire(3)
    f_ln(e[0], e[1], e[3], e[4], e[2], oo[0], oo[1], oo[2])    # ln_fn(x0, x1, a0, a1, w, y0, y1, xn)
    f_rc(oo[2], xs)                                            # keep xm for the router
    aout.release(3)
    ain.release(5)
    for rb in range_(W_ELEMS):
        e = ain.acquire(1)
        f_ra(e, xs, acc, rb)
        ain.release(1)
    o = aout.acquire(1)
    f_rf(acc, o)
    aout.release(1)
