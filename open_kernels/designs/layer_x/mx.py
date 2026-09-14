r"""mx: the MoE block alone -- the second half of lx.py / ax.py's per-token main-core
program (xcommon.moe_body) as its OWN xclbin, for the block prefill route
(OPEN-PREFILL-BATCH): the projections run as GEMMs, the DeltaNet / attention /
norm / router on the host, and this dispatch does the routed + shared experts
for one token. lx1 / ax1 cannot serve there: the main cores run one fixed
program per token and the MoE stream is its second half -- dispatched on its
own it waits on part 0's elements forever.

Routed experts ONLY (nx = NE): the shared expert is 1.97 MB of every token's
17.69 MB stream doing work that is identical across a block, so the block route
runs it once per block as two GEMMs on the host side and folds it into the
residual this dispatch is given. The stream closes on xres + acc instead
(moe_accfin's slot < 0). The whole-layer designs keep all NX slots.

Same main-core kernels, buffers and w / x / y streams as the whole-layer
designs and the same `moe_sequence`, so the driver's moeroute2 patch applies
unchanged. Six buffer arguments in the attention layer's order (pool, xres,
consts, state, act, ptab); only pool, xres, consts and act are touched, the
other two are the dummy-arg idiom dx_attn.py uses to keep the argument
positions the patcher expects. The act / consts offsets differ per layer
type, so MX_KIND=linear | full picks the layout of the stream.

Build (WSL): MX_KIND=linear python build_design.py designs/layer_x/mx.py designs/layer_x/build_mx_linear
             MX_KIND=full   python build_design.py designs/layer_x/mx.py designs/layer_x/build_mx_full
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, InOut, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_HP, A_RES, A_ROUT, A_XM, AA_BYTES, AA_HP, AA_RES, AA_ROUT, AA_XM,  # noqa: E402
                    C_BYTES, C_SGW, CA_BYTES, CA_SGW, KV_BYTES, POOL_BYTES, PTAB_BYTES, STATE_BYTES, R, SPEC)
import xcommon as X  # noqa: E402

KIND = os.environ.get("MX_KIND", "linear")
if KIND not in ("linear", "full"):
    sys.exit(f"mx.py: MX_KIND={KIND!r} (linear | full)")
if R.kind != "moe":
    sys.exit("mx.py: the MoE block of a dense composition is not a thing")
N_CORES, HID = X.N_CORES, X.HID
ELEM = X.ELEM

# the layer type's act / consts / state layout
if KIND == "linear":
    ACT_BYTES, CONSTS_BYTES, STATE_TY_BYTES = A_BYTES, C_BYTES, STATE_BYTES
    XM, ROUT, RES, HP, SGW = A_XM, A_ROUT, A_RES, A_HP, C_SGW
else:
    ACT_BYTES, CONSTS_BYTES, STATE_TY_BYTES = AA_BYTES, CA_BYTES, KV_BYTES
    XM, ROUT, RES, HP, SGW = AA_XM, AA_ROUT, AA_RES, AA_HP, CA_SGW


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def mx(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, ptab: In, *, srchash: CompileTime[int] = 0):
    t = X.types()
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CONSTS_BYTES,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_TY_BYTES,), np.dtype[np.uint8]]   # unused: position only
    act_ty = np.ndarray[(ACT_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]        # unused: position only

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(X.LN), str(X.RT), str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)

    of_w = [ObjectFifo(t["elem"], name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(t["y"], name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(t["x"], name="x", depth=2)

    def main_body(win, xin, yout, *args):
        B, K = X.unpack_args(args)
        X.moe_body(win, xin, yout, B, K, nx=X.NE)

    workers = [Worker(main_body,
                      fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), *X.worker_args(X.core_buffers(t, c), K)],
                      tile=Tile(c, 2), stack_size=0x1800)
               for c in range(N_CORES)]

    def sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, w_prods, x_prod, y_conss):
        # a_state / a_ptab: never touched (see the docstring)
        X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod, y_conss,
                       ACT_BYTES, CONSTS_BYTES, XM, ROUT, RES, HP, SGW, nx=X.NE)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty, ptab_ty,
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = mx
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes(), (HERE / "mx.py").read_bytes()] + X.source_hash_inputs()
                + [(GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode(), KIND.encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
