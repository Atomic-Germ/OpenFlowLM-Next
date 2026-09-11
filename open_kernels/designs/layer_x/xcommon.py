r"""Shared pieces of the whole-layer designs lx / ax: the main cores' GEMV entry
points, the MoE block and the DeltaNet step on those cores (kernels, the core
program fragments, the host-sequence fragments), the norm + router helper core,
and the DMA tap helpers. See lx.py for the design as a whole. The geometry
(cores, widths, element counts, scratch offsets) is the recipe's `Common`
(open_kernels/recipes/qwen36moe.py), computed from the ModelSpec.

Program memory (16 KB per core) shapes everything here: the main core's IRON
program alone was 10 KB with one kernel call site per stage, so the kernels
take ONE scratch buffer each (`ms` for the MoE, `ds` for DeltaNet, fixed
offsets inside), the routed and shared experts share one 9-iteration loop
(the down band law and acc/combine are chosen INSIDE the kernels from the slot
index), and every GEMV shape is one runtime-parameterised entry point.

Main-core streams (all layer types):
  w (10 KB elements from the shim): weights, the MoE header, experts, S slices, DeltaNet records
  x (4 KB elements, broadcast): xn, og (2 elements), xm, the expert hidden h (f32[FF])
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

from layout import R, SPEC, POOL_BYTES, POOL_DOWN, POOL_SHARE_DOWN, POOL_SHARE_GATE, POOL_SHARE_UP

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
ELEM = 4096


def _gemv_prep_entry(k: int) -> Path:
    """Generate the gemv_q4 activation-prep TU for width k (git-ignored), the
    same on-demand entry lm_head_q8 uses: gemv_q4.ensure_prep_entry rewrites it
    only when the text differs, so it is generated at build time, not tracked."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gemv_q4_gen", GEMV / "gemv_q4.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen.ensure_prep_entry(k)

C = R.common
KIND = R.kind                         # "moe" (qwen36moe) | "dense" (qwen35): which tail the layer runs
Q8 = R.q8                             # the projection roles streamed at q8 (OPEN-QUANT-Q8); usually empty
FFN = R.ffn                           # the dense tail's geometry; None on the MoE path
ELN = R.layout.ELN                    # the norm helper's element: HID*2 bytes
NE, NX = C.NE, C.NX                   # routed experts, + the shared one
HID, FF = C.HID, C.FF
TILE = C.TILE
PER_CALL = C.PER_CALL
CALL_BYTES = C.CALL_BYTES             # one w element
STRIPE = C.STRIPE                     # 128 rows x HID (RS=4 band)
HALF = C.HALF                         # 64 rows x HID = UP_ELEMS elements
PAIR = C.PAIR                         # the two chunks of one half at one k-tile = 1 element
DOWN_BAND = C.DOWN_BAND               # 128 rows x FF
UP_BYTES = C.UP_BYTES                 # one expert's up (= gate = down)
N_CORES = C.N_CORES
DOWN_PER_CORE = C.DOWN_PER_CORE
BAND_ROWS = C.BAND_ROWS
BAND16, BAND32 = C.BAND16, C.BAND32   # K=HID / K=2*HID band bytes
N_HDR = C.N_HDR
ROWS_PC, HID_PC = C.ROWS_PC, C.HID_PC # MoE rows per core, hidden per core
OS = ["-Os"]                          # main-core kernels: size over speed (the GEMV is DMA-bound)

# scratch layouts (floats) -- gen_kernels.py writes the same offsets into the kernel TUs
MS_FLOATS = C.MS_FLOATS
DS_FLOATS = C.DS_FLOATS
TAB_BYTES = C.TAB_BYTES               # gemv_q4_tab_bytes(widest K); h's table sits at +H_TAB_OFF
H_TAB_OFF = C.H_TAB_OFF
KWIDE = C.KWIDE                       # the widest K a main core prepares (the og / out projections)


def band_bytes(K: int) -> int:
    return BAND_ROWS * K // 8192 * TILE


def per_band(K: int) -> int:
    """chunks per 64-row band of a K-wide standard-layout matrix (the runtime band law)."""
    return band_bytes(K) // TILE


def n_groups(K: int) -> int:
    """w elements per band."""
    return band_bytes(K) // CALL_BYTES


# ---- the same three, for a projection whose role the container stores at q8: a band is
# still 64 output rows x K, now four 16-row half-tiles per k-tile, so twice the bytes,
# twice the elements and a row split of 4 (designs/gemv_q4/gemv_q8.h). A role that is not
# q8 gets exactly today's numbers -- these helpers are the only branch in the designs.
def role_band_bytes(role: str, K: int) -> int:
    return 2 * band_bytes(K) if role in Q8 else band_bytes(K)


def role_per_band(role: str, K: int) -> int:
    return role_band_bytes(role, K) // TILE


def role_groups(role: str, K: int) -> int:
    return role_band_bytes(role, K) // CALL_BYTES


def role_rs(role: str) -> int:
    return 4 if role in Q8 else 2


# Which GEMV entries this core actually needs. The q4_1 ones go in only while some
# projection still uses them: on a model whose every projection is q8 (the 35B fine-tunes
# are close to it) `gemv_q4_gy` would be dead code inside a 16 KB program.
PROJ_ROLES = ("attn", "linear", "linear_out", "ffn") if KIND == "dense" else ("attn", "linear", "linear_out")
NEED_Q4_GY = any(r not in Q8 for r in PROJ_ROLES)
NEED_Q4_GMS = "ffn" not in Q8              # the dense tail's up | gate bands

# A container that MIXES formats (every Qwen3.5 dense model: linear_out at q8, everything
# else q4_1) needs BOTH GEMV bodies on the main core, and 16 KB of program memory does not
# hold three entries -- the 4B's `lx` overflowed (.claude/plans/q8-hw-results.md section 2).
# So on a mixed spec ONLY, the q4_1 pair folds into one `gemv_q4_gyms` whose destination is
# a runtime argument (dst < 0 -> the band's y element, dst >= 0 -> ms + dst), and the two
# GEMV TUs are compiled -Oz. An all-q4_1 or an all-q8 spec sees exactly the entries, the
# flags and the call sequence it saw before: the DNX_PAD lesson -- what is not identical moves.
MIXED = KIND == "dense" and bool(Q8) and NEED_Q4_GY and NEED_Q4_GMS
GEMV_OS = ["-Oz"] if MIXED else OS         # size over speed, harder, on the crowded core only


def bt(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def half_tap(off: int) -> TensorAccessPattern:
    """One 64-row half of an RS=4 stripe in the pool: chunk pairs at every k-tile (HID/256 x 10240 B,
    stride 20480), as three real DMA dims (the BD's highest dim is a repeat count, its length
    covers only the lowest three, the innermost wrap is < 4096 B)."""
    return TensorAccessPattern((1, POOL_BYTES), off, [1, HID // 256, 4, PAIR // 4], [0, 2 * PAIR, PAIR // 4, 1])


def types():
    t = {}
    t["elem"] = np.ndarray[(CALL_BYTES,), np.dtype[np.uint8]]
    t["x"] = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]      # one 4 KB act element (bf16 view)
    t["y"] = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]    # one y element
    t["tab"] = np.ndarray[(TAB_BYTES,), np.dtype[np.uint8]]
    t["ms"] = np.ndarray[(MS_FLOATS,), np.dtype[np.float32]]
    t["ds"] = np.ndarray[(DS_FLOATS,), np.dtype[np.float32]]
    return t


# The DeltaNet kernels' element geometry: rows of S per streamed weight element, and
# nothing else. dnx.h's `kPad` is the hi/lo record stride inside `ds` (a fixed 160, set by
# the DS_KHL / DS_QHL slot geometry), NOT `Common.DN_PAD` -- DN_PAD is the padded S row
# count of the state buffer, 140 for the 27B, and passing it as -DDNX_PAD rebuilt every
# dnx_* object in the shipped kernels. dnx.h's default IS the 27B's 20, so the MoE path
# compiles with exactly the flags it always did.
DNX_ROWS_DEFAULT = 20                 # dnx.h's #ifndef DNX_ROWS value
DN_FLAGS = [] if C.DN_ROWS in (0, DNX_ROWS_DEFAULT) else [f"-DDNX_ROWS={C.DN_ROWS}"]


def kernels(inc, t):
    e, x, y, tab, ms, ds = t["elem"], t["x"], t["y"], t["tab"], t["ms"], t["ds"]
    i32 = np.int32
    nb = ELEM // 2 // 32                                            # bf16 blocks of 32 per 4 KB element

    def ef(sym, args, flags=None):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=OS if flags is None else flags)

    def dnf(sym, args):
        return ef(sym, args, OS + DN_FLAGS)

    def add_q8(k):
        """The q8 GEMV entries. Instantiated ONLY when a role is q8: an ExternalFunction
        that exists changes the build, so a q4_1 model must see the dictionary it always
        saw (the DNX_PAD lesson -- a flag equal to a default is not always a no-op)."""
        if Q8:
            k["gy8"] = ef("gemv_q8_gy", [e, tab, y, i32, i32, i32], GEMV_OS)
        if "ffn" in Q8:
            k["gms8"] = ef("gemv_q8_gms", [e, tab, ms, i32, i32, i32], GEMV_OS)
        return k

    if KIND == "dense":
        k = {}
        if MIXED:
            # one entry for both q4_1 destinations (gen_kernels.q4_gyms)
            k["gyms"] = ef("gemv_q4_gyms", [e, tab, y, ms, i32, i32, i32], GEMV_OS)
        # A projection band into its y element (the same entry as the MoE path).
        if NEED_Q4_GY and not MIXED:
            k["gy"] = ef("gemv_q4_gy", [e, tab, y, i32, i32, i32])
        # The dense FFN tail (designs/dense/dx.py's kernels, generated into this design):
        # an up | gate band into the silu scratch, act(gate) * up, and the two element-indexed
        # activation preps (the core loops over 4 KB elements; the kernel derives the blocks).
        if NEED_Q4_GMS and not MIXED:
            k["gms"] = ef("gemv_q4_gms", [e, tab, ms, i32, i32, i32])
        k["act"] = ef("dense_act", [ms, y])
        k["prep"] = ef("dense_prep", [x, tab, i32, i32])
        k["prepf"] = ef("dense_prep_f32", [x, tab, i32, i32])
        k["vcopy"] = dnf("dnx_vcopy", [e, ds])
        k["p1"] = dnf("dnx_pass1", [e, ds, i32])
        k["delta"] = dnf("dnx_delta", [ds])
        k["row"] = dnf("dnx_row", [e, ds, y, i32, i32])
        k["ofin"] = dnf("dnx_ofin", [ds, y, i32])
        return add_q8(k)
    k = {}
    # GEMVs (runtime group / band law): projections into a y element; MoE up/gate and down into ms
    if NEED_Q4_GY:
        k["gy"] = ef("gemv_q4_gy", [e, tab, y, i32, i32, i32])      # (t, tab, ye, group, per_band, rs)
    k["gup"] = ef("gemv_q4_gup", [e, tab, ms, i32, i32])            # (t, tab, ms, group, band)  u | g
    k["gdown"] = ef("gemv_q4_gdown", [e, tab, ms, i32, i32])        # (t, tab, ms, j, slot)      routed / shared law
    # activation tables: x (one element, K = HID) and the two-element og (K = KWIDE)
    k["prep2048"] = ExternalFunction(f"gemv_q4_prep_k{HID}", source_file=str(_gemv_prep_entry(HID)),
                                     arg_types=[x, tab], include_dirs=inc, compile_flags=OS)
    k["prep4096a"] = ef(f"gemv_q4_prep_k{KWIDE}_b0n{nb}", [x, tab])
    k["prep4096b"] = ef(f"gemv_q4_prep_k{KWIDE}_b{nb}n{nb}", [x, tab])
    k["prepf"] = ef("gemv_q4_prep_h", [x, tab])                     # h (f32[FF] in the element) -> tab + H_TAB_OFF
    # MoE
    k["hdr"] = ef("moe_hdr2", [e, x, ms, i32])
    k["silu"] = ef("moe_silu32", [ms, y])
    k["accfin"] = ef("moe_accfin", [ms, i32])
    k["out"] = ef("moe_out", [ms, y, i32])
    # DeltaNet
    k["vcopy"] = dnf("dnx_vcopy", [e, ds])
    k["p1"] = dnf("dnx_pass1", [e, ds, i32])
    k["delta"] = dnf("dnx_delta", [ds])
    k["row"] = dnf("dnx_row", [e, ds, y, i32, i32])
    k["ofin"] = dnf("dnx_ofin", [ds, y, i32])
    return add_q8(k)


KNAMES_MOE = ("gy", "gup", "gdown", "prep2048", "prep4096a", "prep4096b", "prepf", "hdr", "silu", "accfin", "out",
              "vcopy", "p1", "delta", "row", "ofin")
KNAMES_DENSE = ("gy", "gms", "act", "prep", "prepf", "vcopy", "p1", "delta", "row", "ofin")
KNAMES_Q8 = (("gy8",) if Q8 else ()) + (("gms8",) if "ffn" in Q8 else ())


def _knames() -> tuple:
    """The kernels a main core holds, in the order `kernels()` builds them. On a mixed
    spec the folded entry stands where the pair stood; every other spec keeps its order."""
    base = KNAMES_MOE if KIND == "moe" else KNAMES_DENSE
    ns = [n for n in base if (n != "gy" or NEED_Q4_GY) and (n != "gms" or NEED_Q4_GMS)]
    if MIXED:
        ns = ["gyms"] + [n for n in ns if n not in ("gy", "gms")]
    return tuple(ns) + KNAMES_Q8


KNAMES = _knames()
BNAMES = ("tab", "ms", "ds")


def core_buffers(t, c):
    return dict(tab=Buffer(t["tab"], name=f"tab{c}"), ms=Buffer(t["ms"], name=f"ms{c}"), ds=Buffer(t["ds"], name=f"ds{c}"))


def worker_args(B, K):
    return [*[B[n] for n in BNAMES], *[K[n] for n in KNAMES]]


def unpack_args(args):
    B = dict(zip(BNAMES, args[:len(BNAMES)]))
    K = dict(zip(KNAMES, args[len(BNAMES):]))
    return B, K


def gemv_bands(win, yout, tab, gy, nbands, ngroups, per_band, rs, ms=None):
    """nbands bands of ngroups elements each against the table, one y element per band.
    `ms` is passed only by the folded mixed-format entry, which takes both destinations and
    picks between them with dst (-1 = this band's y element)."""
    for _ in range_(nbands):
        ye = yout.acquire(1)
        for g in range_(ngroups):
            we = win.acquire(1)
            if ms is None:
                gy(we, tab, ye, g, per_band, rs)
            else:
                gy(we, tab, ye, ms, g, per_band, -1)
            win.release(1)
        yout.release(1)


def role_gemv_bands(win, yout, B, K, role, nbands, KK):
    """`nbands` bands of a KK-wide projection of `role`, at that role's weight format.
    A q4_1 role runs exactly the call it always ran, unless this core carries the folded
    entry -- then the same band goes through `gemv_q4_gyms` with dst = -1."""
    tab = B["tab"]
    if role in Q8:
        gemv_bands(win, yout, tab, K["gy8"], nbands, role_groups(role, KK), role_per_band(role, KK), 4)
    elif MIXED:
        gemv_bands(win, yout, tab, K["gyms"], nbands, n_groups(KK), per_band(KK), 2, B["ms"])
    else:
        gemv_bands(win, yout, tab, K["gy"], nbands, n_groups(KK), per_band(KK), 2)


# ---- the dense FFN tail on one main core (ffn="dense"; designs/dense/dx.py steps 6-7)
def prep_bands(win, xin, yout, B, K, kk, n_elems, nbands, role="attn"):
    """Prepare the KK-wide activation from `n_elems` 4 KB x elements (the kernel derives
    each element's block range from its index, so no arithmetic on the loop variable),
    then run `nbands` bands of it at `role`'s weight format. The dense counterpart of the
    MoE's fixed prep entries."""
    tab = B["tab"]
    xe = xin.acquire(n_elems)
    if n_elems == 1:
        K["prep"](xe, tab, kk, 0)
    else:
        for i in range(n_elems):
            K["prep"](xe[i], tab, kk, i)
    role_gemv_bands(win, yout, B, K, role, nbands, kk)
    xin.release(n_elems)


def ffn_body(win, xin, yout, B, K):
    """up | gate per 64-row band into `ms`, act(gate) * up out through y, then the down
    GEMV against h (assembled in DDR from the cores' bands, read back as f32 elements)."""
    tab, ms = B["tab"], B["ms"]
    me = xin.acquire(FFN.XM_ELEMS)
    if FFN.XM_ELEMS == 1:
        K["prep"](me, tab, HID, 0)
    else:
        for i in range(FFN.XM_ELEMS):
            K["prep"](me[i], tab, HID, i)
    if "ffn" in Q8:
        gms, pb_h, ng_h = K["gms8"], role_per_band("ffn", HID), role_groups("ffn", HID)
    elif MIXED:
        gms, pb_h, ng_h = K["gyms"], per_band(HID), n_groups(HID)
    else:
        gms, pb_h, ng_h = K["gms"], per_band(HID), n_groups(HID)

    def band(we, ye, g, dst):
        """One up | gate band into ms + dst. The folded entry takes the y pointer too, so
        on a mixed core the band's y element is acquired first -- the shape `gemv_bands`
        already runs (acquire y, stream the w elements, release)."""
        if MIXED:
            gms(we, tab, ye, ms, g, pb_h, dst)
        else:
            gms(we, tab, ms, g, pb_h, dst)

    for _ in range_(FFN.UP_PC):
        ye = yout.acquire(1) if MIXED else None
        for g in range_(ng_h):
            we = win.acquire(1)
            band(we, ye, g, C.MS_U)
            win.release(1)
        for g in range_(ng_h):
            we = win.acquire(1)
            band(we, ye, g, C.MS_G)
            win.release(1)
        if not MIXED:
            ye = yout.acquire(1)
        K["act"](ms, ye)
        yout.release(1)
    xin.release(FFN.XM_ELEMS)
    for i in range_(FFN.H_ELEMS):
        he = xin.acquire(1)
        K["prepf"](he, tab, FF, i)
        xin.release(1)
    role_gemv_bands(win, yout, B, K, "ffn", FFN.DOWN_PC, FF)


def ffn_sequence(pipe_w, pipe_x, pipe_y, a_pool, a_act, w_prods, x_prod, y_conss,
                 A_BYTES, A_XM, A_H, A_OUT2, POOL_UP, POOL_GATE, POOL_DOWN_FFN):
    """Host side of ffn_body. The h drains are issued before the per-band weight fills so a
    core is never blocked on a full y fifo while the host is still pacing its w stream."""
    bb_h, bb_f, yb = role_band_bytes("ffn", HID), role_band_bytes("ffn", FF), BAND_ROWS * 4
    pipe_x.fill(x_prod, a_act, bt(A_BYTES, A_XM, FFN.XM_ELEMS * ELEM))
    for c in range(N_CORES):
        pipe_y.drain(y_conss[c], a_act, bt(A_BYTES, A_H + c * FFN.UP_PC * yb, FFN.UP_PC * yb))
    for j in range(FFN.UP_PC):
        for c in range(N_CORES):
            pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_UP + (c * FFN.UP_PC + j) * bb_h, bb_h))
            pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_GATE + (c * FFN.UP_PC + j) * bb_h, bb_h))
    pipe_y.finish(*y_conss)                                   # h is in DDR
    pipe_x.fill(x_prod, a_act, bt(A_BYTES, A_H, FFN.H_ELEMS * ELEM))
    for c in range(N_CORES):
        pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_DOWN_FFN + c * FFN.DOWN_PC * bb_f, FFN.DOWN_PC * bb_f))
        pipe_y.drain(y_conss[c], a_act, bt(A_BYTES, A_OUT2 + c * FFN.DOWN_PC * yb, FFN.DOWN_PC * yb))


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
    for e in range_(NX):                          # NE routed slots, then the shared expert
        for b in range_(2):                       # u then g, HID_PC rows each, UP_ELEMS elements per band
            for g in range_(C.UP_ELEMS):
                we = win.acquire(1)
                K["gup"](we, tab, ms, g, b)
                win.release(1)
        he = yout.acquire(1)
        K["silu"](ms, he)                         # this core's rows of the hidden, f32, to DDR
        yout.release(1)
        hh = ain.acquire(1)
        K["prepf"](hh, tab)                       # the whole h back, -> its table (tab + H_TAB_OFF)
        ain.release(1)
        for j in range_(C.DOWN_ELEMS):            # the core's ROWS_PC down rows: DOWN_ELEMS elements either way
            we = win.acquire(1)
            K["gdown"](we, tab, ms, j, e)
            win.release(1)
        K["accfin"](ms, e)                        # routed: acc += w[e] y; shared: out = xres + acc + gate y
    for j in range_(C.OUT_ELEMS):
        ye = yout.acquire(1)
        K["out"](ms, ye, j)
        yout.release(1)


def moe_sequence(pipe_w, pipe_x, pipe_y, a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                 A_BYTES, C_BYTES, A_XM, A_ROUT, A_RES, A_HP, C_SGW):
    """Host sequence of the MoE block (one instruction-stream part). Routed slot j's fills carry
    placeholder pool offsets (expert j); moeroute2 rewrites them from the router output."""
    spp, cps = C.STRIPES_PER_PROJ, C.CORES_PER_STRIPE
    pipe_x.fill(x_prod, a_act, bt(A_BYTES, A_XM, ELEM))
    for c in range(N_CORES):
        pipe_w.fill(w_prods[c], a_act, bt(A_BYTES, A_ROUT, CALL_BYTES))
        pipe_w.fill(w_prods[c], a_consts, bt(C_BYTES, C_SGW, CALL_BYTES))
        pipe_w.fill(w_prods[c], a_act, bt(A_BYTES, A_RES + c * ROWS_PC * 4, CALL_BYTES))
    for e in range(NX):
        for c in range(N_CORES):
            if e < NE:
                up = (2 * spp * e + 2 * (c // cps)) * STRIPE + (c % cps) * PAIR
                pipe_w.fill(w_prods[c], a_pool, half_tap(up))
                pipe_w.fill(w_prods[c], a_pool, half_tap(up + STRIPE))
            else:
                pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_SHARE_UP + c * HALF, HALF))
                pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_SHARE_GATE + c * HALF, HALF))
            pipe_y.drain(y_conss[c], a_act, bt(A_BYTES, A_HP + c * HID_PC * 4, HID_PC * 4))
        pipe_y.finish(*y_conss)                           # the hidden parts are in DDR
        pipe_x.fill(x_prod, a_act, bt(A_BYTES, A_HP, ELEM))
        for c in range(N_CORES):
            if e < NE:
                pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_DOWN + e * UP_BYTES + c * DOWN_PER_CORE * DOWN_BAND,
                                                   DOWN_PER_CORE * DOWN_BAND))
            else:
                pipe_w.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_SHARE_DOWN + c * DOWN_PER_CORE * DOWN_BAND,
                                                   DOWN_PER_CORE * DOWN_BAND))
    for c in range(N_CORES):
        pipe_y.drain(y_conss[c], c_xres, bt(HID, c * ROWS_PC, ROWS_PC))   # the block output = the new residual
    pipe_w.finish()
    pipe_x.finish()
    pipe_y.finish()


# ---- DeltaNet on the main cores (dnx.h): S slices ride the w stream, S' rows leave through y
DN_ROWS, DN_SLICES, DN_HEADS_PC = C.DN_ROWS, C.DN_SLICES, C.DN_HEADS_PC


def dn_body(win, yout, B, K):
    """This core's heads: the record (copied out of its element: release() frees the OLDEST held
    element), DN_SLICES slices (pass 1), delta, DN_SLICES slices x 2*DN_ROWS half rows (pass 2, into
    y elements), o."""
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
    rec, ohb = R.linear.RECORD_BYTES, R.linear.O_HEAD_BYTES
    for c in range(N_CORES):
        for h in range(DN_HEADS_PC):
            hd = c * DN_HEADS_PC + h
            pipe_w.fill(w_prods[c], a_act, bt(A_BYTES, A_VEC + hd * rec, CALL_BYTES))
            pipe_w.fill(w_prods[c], a_state, bt(STATE_BYTES, STATE_S_OFF + hd * S_HEAD_BYTES, S_HEAD_BYTES))
            pipe_y.drain(y_conss[c], a_state, bt(STATE_BYTES, STATE_S_OFF + hd * S_HEAD_BYTES, S_HEAD_BYTES))
            pipe_y.drain(y_conss[c], a_act, bt(A_BYTES, A_O + hd * ohb, ohb))
            pipe_w.fill(w_prods[c], a_state, bt(STATE_BYTES, STATE_S_OFF + hd * S_HEAD_BYTES, S_HEAD_BYTES))


# ---- the norm + router helper core (both layer types): ln_nr -> ln(+residual) -> router
LN = HERE.parent / "ln"
LINL = HERE.parent / "lin_layer"
RT = HERE.parent / "router"
W_ELEMS = C.W_ELEMS                   # router W as elements of 4 KB


# LN_N / LN_EPS default to the 27B's in ln.h and ln_nr.cc, so the MoE path compiles with
# exactly the flags it always did; a different width or epsilon passes them.
LN_FLAGS = ([] if (HID, SPEC.norm_eps) == (2048, 1e-6)
            else [f"-DLN_N={HID}", f"-DLN_EPS={SPEC.norm_eps:g}f"])


def ln_types():
    """The norm helper's element. The MoE designs hold five inputs and three outputs of it
    at once (`ln_fn`), which only fits at 4 KB; the dense composition sizes it ELN = HID*2
    (8 KB at HID 4096) and splits the outputs one per call, as designs/dense/dx.py does."""
    if KIND == "dense":
        return dict(u8_ln=np.ndarray[(ELN,), np.dtype[np.uint8]])
    return dict(u8_4k=np.ndarray[(ELEM,), np.dtype[np.uint8]], xb=np.ndarray[(HID,), np.dtype[bfloat16]],
                racc=np.ndarray[(SPEC.num_experts,), np.dtype[np.float32]])


def ln_kernels(inc, t):
    if KIND == "dense":
        u, i32 = t["u8_ln"], np.int32
        return {
            "ln_nr": ExternalFunction("ln_nr", source_file=str(LINL / "ln_nr.cc"), arg_types=[u] * 4,
                                      include_dirs=inc, compile_flags=LN_FLAGS),
            "ln_y": ExternalFunction("ln_y", source_file=str(LN / "ln_y.cc"), arg_types=[u] * 5 + [i32],
                                     include_dirs=inc, compile_flags=LN_FLAGS),
            "ln_xn": ExternalFunction("ln_xn", source_file=str(LN / "ln_xn.cc"), arg_types=[u] * 6,
                                      include_dirs=inc, compile_flags=LN_FLAGS),
        }
    u = t["u8_4k"]
    k = {}
    k["ln_nr"] = ExternalFunction("ln_nr", source_file=str(LINL / "ln_nr.cc"), arg_types=[u] * 4, include_dirs=inc)
    k["ln"] = ExternalFunction("ln_fn", source_file=str(LN / "ln.cc"), arg_types=[u] * 8, include_dirs=inc)
    k["rcopy"] = ExternalFunction("router_copy_x", source_file=str(RT / "router_copy.cc"), arg_types=[u, t["xb"]], include_dirs=inc)
    k["racc"] = ExternalFunction("router_acc", source_file=str(RT / "router.cc"), arg_types=[u, t["xb"], t["racc"], np.int32], include_dirs=inc)
    k["rfin"] = ExternalFunction("router_fin", source_file=str(RT / "router_fin.cc"), arg_types=[t["racc"], u], include_dirs=inc)
    return k


def ln_body(ain, aout, f_nr, f_lny, f_lnx):
    """The dense composition's norm helper: the same three stages the MoE's ln_router_body
    runs minus the router, with the residual+norm's outputs one element per call.

      1. layer-entry norm            [x0 x1 lnw] -> [xn]
      2. res = x + attn_out, xm = post_attention_norm(res)   [x0 x1 w a0 a1] -> [y0] [y1] [xm]
      3. xres = res + ffn_out                                 [x0 x1 w a0 a1] -> [y0] [y1] [junk]
    """
    e = ain.acquire(3)
    o = aout.acquire(1)
    f_nr(e[0], e[1], e[2], o)
    aout.release(1)
    ain.release(3)
    for _ in range(2):
        e = ain.acquire(5)
        for i in range(2):
            o = aout.acquire(1)
            f_lny(e[0], e[1], e[3], e[4], o, i)
            aout.release(1)
        o = aout.acquire(1)
        f_lnx(e[0], e[1], e[3], e[4], e[2], o)      # stage 3's xn is junk (nothing reads it)
        aout.release(1)
        ain.release(5)


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


def source_hash_inputs():
    """The recipe sources, for the designs' srchash (a recipe change re-jits)."""
    return sorted(f.read_bytes() for f in (HERE.parent.parent / "recipes").glob("*.py"))
