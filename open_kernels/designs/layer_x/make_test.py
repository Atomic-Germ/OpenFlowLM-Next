r"""Test vectors for lx (the whole linear-attention layer + MoE) from layer_chain's
layer-0 inputs and moe_chain's MoE references (no model needed; runs on Windows):

    run lx0 pool xres consts state act        ln -> qkv|z -> glue -> DeltaNet -> post -> out -> ln+res -> router
    moeroute2 lx1 act <rout idx offset>
    run lx1 pool xres consts state act        the MoE block -> xres

The weight arg is the captured layer-0 pool (qkv/z/experts at their offsets); the
out projection, router W and sgw are in consts. References: layer_chain's fp64
replica (xn, xres after attention, xm, S, conv state), moe_chain's routing and
block output (the layer output).

    python make_test.py ; run_kernel run.cfg ; python compare.py

The pool is the captured layer-0 pool ($OPEN_KERNELS_CAPS/m0d/blob_536870912_*.bin);
paths in run.cfg are relative to this directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LC = HERE.parent / "layer_chain"
MC = HERE.parent / "moe_chain"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402
from layout import (A_BYTES, A_ROUT, C_BYTES, C_LNW, C_NW, C_POSTLN, C_RW, C_SGW, C_SIDE, C_WOUT,  # noqa: E402
                    GLUE_SIDE_BYTES, POOL_BYTES, S_HEAD_BYTES, STATE_BYTES, STATE_S_OFF)

POOL_CAP = "m0d/blob_536870912_836fd8e49f35a0b6.bin"


def main() -> int:
    consts = np.zeros(C_BYTES, np.uint8)
    consts[C_LNW:C_LNW + 4096] = np.fromfile(LC / "lnw.bin", np.uint8)
    consts[C_SIDE:C_SIDE + GLUE_SIDE_BYTES] = np.fromfile(LC / "side_glue.bin", np.uint8)[4096:4096 + GLUE_SIDE_BYTES]
    consts[C_NW:C_NW + 4096] = np.fromfile(LC / "nw.bin", np.uint8)
    consts[C_POSTLN:C_POSTLN + 4096] = np.fromfile(LC / "postln.bin", np.uint8)
    rw = np.fromfile(MC / "router_w.bin", np.uint8)
    assert len(rw) == 1048576, len(rw)
    consts[C_RW:C_RW + 1048576] = rw
    consts[C_SGW:C_SGW + 4096] = np.fromfile(MC / "sgw.bin", np.uint8)[:4096]
    wout = np.fromfile(LC / "w_out.bin", np.uint8)
    wout = wout[:10485760]                     # [2048, 4096] q4 = 5242880 B; the region is oversized
    consts[C_WOUT:C_WOUT + len(wout)] = wout
    (HERE / "consts.bin").write_bytes(consts.tobytes())
    (HERE / "xres.bin").write_bytes((LC / "x_res.bin").read_bytes())
    state = np.zeros(STATE_BYTES, np.uint8)
    state[:49152] = np.fromfile(LC / "state.bin", np.uint8)[:49152]
    s_in = np.fromfile(LC / "s_in.bin", np.uint8).reshape(32, 128 * 512)
    sp = state[STATE_S_OFF:].reshape(32, S_HEAD_BYTES)
    sp[:, :128 * 512] = s_in                                     # rows 128..139 stay zero
    (HERE / "state.bin").write_bytes(state.tobytes())
    for n in ("ref_xn", "ref_xres", "ref_xm", "ref_S", "ref_cs"):
        (HERE / f"{n}.bin").write_bytes((LC / f"{n}.bin").read_bytes())
    (HERE / "ref_out.bin").write_bytes((MC / "ref_out.bin").read_bytes())
    (HERE / "ref_rout.bin").write_bytes((MC / "y_rout.bin").read_bytes())
    runs = [
        "run lx0 pool xres consts state act",
        f"moeroute2 lx1 act {A_ROUT + 1024}",
        "run lx1 pool xres consts state act",
    ]
    reload = ["load xres xres.bin", "load state state.bin"]
    cfg = [
        "device",
        "xclbin X build_lx0/final.xclbin",
        "kernelx lx0 X build_lx0/insts.bin", "kernelx lx1 X build_lx1/insts.bin",
        f"buf pool {POOL_BYTES} {FX.caps_cfg(POOL_CAP)}",
        "buf xres 8192 xres.bin",
        f"buf consts {C_BYTES} consts.bin",
        f"buf state {STATE_BYTES} state.bin",
        f"buf act {A_BYTES}",
        *runs,
        f"dump act y_act.bin {A_BYTES}",
        "dump xres y_xres.bin 8192",
        f"dump state y_state.bin {STATE_BYTES}",
        *reload, *runs, *reload, *runs,
        "",
    ]
    (HERE / "run.cfg").write_text("\n".join(cfg), newline="\n")
    print("wrote consts.bin xres.bin state.bin run.cfg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
